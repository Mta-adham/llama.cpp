// Port of journey-video/review/llamacpp-config-discovery/probe.cpp: greedy tokens, text, TTFT and
// process peak RSS. Changes: Windows peak-memory query, and the chat wrapper can be overridden with
// EDGE_TEMPLATE ("{prompt}" is replaced) so non-Gemma models can be probed. The default wrapper is
// the original Gemma 4 one.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
static void fill(llama_batch_ext *b,const llama_token *t,int n,int start){
 llama_batch_ext_clear(b);for(int i=0;i<n;i++){int k=llama_batch_ext_add_token(b,0,t[i]);llama_pos pos=start+i;llama_batch_ext_set_pos(b,k,&pos);}llama_batch_ext_set_output_logits(b,n-1,true);
}
static void jsonstr(const std::string &s){putchar('"');for(unsigned char c:s){if(c=='"'||c=='\\')printf("\\%c",c);else if(c<32)printf("\\u%04x",c);else putchar(c);}putchar('"');}
static int setting(const char *k,int d){const char *v=getenv(k);return v?atoi(v):d;}
static double peak_rss_mib(){
#ifdef _WIN32
 PROCESS_MEMORY_COUNTERS pmc;return GetProcessMemoryInfo(GetCurrentProcess(),&pmc,sizeof(pmc))?pmc.PeakWorkingSetSize/1048576.0:0;
#else
 struct rusage usage;getrusage(RUSAGE_SELF,&usage);return usage.ru_maxrss/1024.0;
#endif
}
static std::string wrap(const std::string &p){
 const char *t=getenv("EDGE_TEMPLATE");if(!t)return "<|turn>user\n"+p+"<turn|>\n<|turn>model\n";
 std::string s=t;size_t i=s.find("{prompt}");if(i!=std::string::npos)s.replace(i,8,p);return s;
}
int main(int argc,char **argv){
 if(argc!=5)return 2;int threads=atoi(argv[3]);bool latency=std::string(argv[4])=="latency";
 std::ifstream in(argv[2]);if(!in)return 3;std::vector<std::string> prompts;std::string line;
 while(std::getline(in,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();if(!line.empty())prompts.push_back(line);}if(prompts.empty())return 4;
 llama_backend_init();auto load=Clock::now();auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.load_mode=(llama_load_mode)setting("EDGE_LOAD",-1);mp.use_extra_bufts=setting("EDGE_REPACK",1);mp.lazy_mode=(llama_lazy_mode)setting("EDGE_LAZY",1);
 auto *model=llama_model_load_from_file(argv[1],mp);if(!model)return 5;auto *vocab=llama_model_get_vocab(model);
 auto cp=llama_context_default_params();cp.n_ctx=setting("EDGE_CTX",2048);cp.n_batch=setting("EDGE_BATCH",512);cp.n_ubatch=setting("EDGE_UBATCH",512);cp.n_threads=threads;cp.n_threads_batch=setting("EDGE_THREADS_BATCH",threads);cp.flash_attn_type=setting("EDGE_FA",0)?LLAMA_FLASH_ATTN_TYPE_ENABLED:LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.type_k=(ggml_type)setting("EDGE_K",1);cp.type_v=(ggml_type)setting("EDGE_V",1);
 auto *ctx=llama_init_from_model(model,cp);if(!ctx)return 6;double load_ms=ms(load);
 auto *sampler=llama_sampler_init_greedy();auto *batch=llama_batch_ext_init(ctx);
 int repeats=latency?3:(int)prompts.size();
 for(int r=0;r<repeats;r++){
  llama_memory_clear(llama_get_memory(ctx),true);llama_sampler_reset(sampler);
  std::string prompt=wrap(prompts[latency?0:r]);
  int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);if(n<1||n+64>(int)cp.n_ctx)return 7;
  std::vector<llama_token> ids(n);if(llama_tokenize(vocab,prompt.data(),prompt.size(),ids.data(),n,true,true)!=n)return 8;
  auto start=Clock::now();
  for(int j=0;j<n;j+=cp.n_batch){int sz=std::min((int)cp.n_batch,n-j);fill(batch,ids.data()+j,sz,j);if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch))return 9;}
  std::vector<int> output;std::string txt;double ttft=0;auto decode_start=Clock::now();int pos=n;
  for(int j=0;j<setting("EDGE_TOKENS",64);j++){
   if(j){if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch))return 10;}
   llama_token t=llama_sampler_sample(sampler,ctx,-1);if(!j){ttft=ms(start);decode_start=Clock::now();}
   output.push_back(t);char buf[1024];int sz=llama_token_to_piece(vocab,t,buf,sizeof(buf),0,false);if(sz<0)return 11;txt.append(buf,sz);
   if(llama_vocab_is_eog(vocab,t))break;fill(batch,&t,1,pos++);
  }
  double decode_ms=ms(decode_start);
  printf("{\"index\":%d,\"prompt_tokens\":%d,\"ttft_ms\":%.6f,\"decode_ms\":%.6f,\"decode_tokens\":%zu,\"load_ms\":%.6f,\"peak_rss_mib\":%.6f,\"text\":",r,n,ttft,decode_ms,output.size()-1,load_ms,peak_rss_mib());jsonstr(txt);printf(",\"tokens\":[");for(size_t j=0;j<output.size();j++)printf("%s%d",j?",":"",output[j]);printf("]}\n");fflush(stdout);
 }
 llama_batch_ext_free(batch);llama_sampler_free(sampler);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
