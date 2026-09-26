// Effective clock: a dependent add chain retires exactly one per cycle.
#include <stdio.h>
#include <stdint.h>
#include <time.h>
int main(void){
  volatile uint64_t sink=0; uint64_t x=1; const uint64_t N=2000000000ULL;
  struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
  for(uint64_t i=0;i<N;i++) x+=i;   // loop-carried dependency: 1 cycle each
  clock_gettime(CLOCK_MONOTONIC,&b); sink=x;
  double s=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
  printf("effective %.2f GHz  (%.3f s for %llu dependent adds) sink=%llu\n",
         N/s/1e9, s, (unsigned long long)N, (unsigned long long)sink);
  return 0;
}
