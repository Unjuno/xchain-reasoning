// SPDX-License-Identifier: Apache-2.0
// Fixed-depth control without decision-reversal features or subset bookkeeping.
#include <algorithm>
#include <cmath>
#include <vector>
#include <new>
extern "C" int xchain_fixed(int n,int e,int b,int q,const int* src,const int* dst,
    const int* rev,const double* jc,const double* fields,int depth,double* out) {
  if(n<1 || n>256 || e<1 || e>65536 || b<1 || q<0 || q>=n || depth<2 || depth>256) return 1;
  for(int a=0;a<e;++a)
    if(src[a]<0 || src[a]>=n || dst[a]<0 || dst[a]>=n || rev[a]<0 || rev[a]>=e || !std::isfinite(jc[a])) return 2;
  try {
    std::vector<double> tj(e),msg(e),next(e),total(n);
    std::vector<int> qin;
    for(int a=0;a<e;++a){tj[a]=std::tanh(jc[a]);if(dst[a]==q) qin.push_back(a);}
    for(int s=0;s<b;++s){
      const double* h=fields+static_cast<std::size_t>(s)*n;
      std::fill(msg.begin(),msg.end(),0.0);
      for(int t=0;t<depth;++t){
        std::fill(total.begin(),total.end(),0.0);
        for(int a=0;a<e;++a)total[dst[a]]+=msg[a];
        for(int a=0;a<e;++a){
          double cavity=(h[src[a]]+total[src[a]])-msg[rev[a]];
          double z=std::max(-1.0+1e-14,std::min(1.0-1e-14,tj[a]*std::tanh(cavity)));
          next[a]=std::atanh(z);
        }
        msg.swap(next);
      }
      double answer=h[q];for(int a:qin)answer+=msg[a];out[s]=answer;
    }
  }catch(const std::bad_alloc&){return 3;}catch(...){return 4;}
  return 0;
}
