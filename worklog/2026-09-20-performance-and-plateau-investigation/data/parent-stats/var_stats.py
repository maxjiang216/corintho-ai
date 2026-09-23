import statistics as st, math, sys
def load(f):
    d={}
    for l in open(f):
        a,s,e,t,r=l.split(); d.setdefault(a,{})[s]=(float(e),int(t),int(r))
    return d
for f in ('var_stub.txt','var_nn.txt'):
    d=load(f); print('==', f, ' identical workloads:', all(d[a][s][1:]==d['HEAD'][s][1:] for a in d for s in d[a]))
    for base in ('HEAD','C0'):
        for a in ('C0','A','B'):
            if a==base: continue
            rel=[d[a][s][0]/d[base][s][0]-1 for s in d[base]]
            m=st.mean(rel); se=st.stdev(rel)/math.sqrt(len(rel))
            print(f"  {a:2s} vs {base:4s}: {100*m:+6.2f}%  (95% CI {100*(m-1.96*se):+.2f} .. {100*(m+1.96*se):+.2f})")
r={}
for l in open('var_rss.txt'):
    a,k=l.split(); r.setdefault(a,[]).append(int(k)/1024)
print('== peak RSS, 2000 games x 8 threads (MB, 2 runs)')
for a in ('HEAD','C0','A','B'): print(f"  {a:4s} {' '.join(f'{x:6.0f}' for x in r[a])}   vs HEAD {100*(min(r[a])/min(r['HEAD'])-1):+.1f}%")
