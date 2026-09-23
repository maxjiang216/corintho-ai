import statistics as st, math, sys
d={}
for l in open(sys.argv[1]):
    a,s,e,t,r=l.split(); d.setdefault(a,{})[s]=(float(e),int(t),int(r))
b,f=d['before'],d['after']
def welch(name, fn):
    x=[fn(*b[s]) for s in b]; y=[fn(*f[s]) for s in f]
    se=math.sqrt(st.variance(x)/len(x)+st.variance(y)/len(y))
    print(f"{name:15s} before {st.mean(x):9.2f}  after {st.mean(y):9.2f}  ({100*(st.mean(y)/st.mean(x)-1):+.2f}%, t = {(st.mean(y)-st.mean(x))/se:+.2f})")
welch('turns/game', lambda e,t,r: t/20)
welch('requests/turn', lambda e,t,r: r/t)
rel=[(f[s][0]/f[s][2])/(b[s][0]/b[s][2])-1 for s in b]
m=st.mean(rel); se=st.stdev(rel)/math.sqrt(len(rel))
print(f"ns/request paired {100*m:+.2f}%  (95% CI {100*(m-1.96*se):+.2f} .. {100*(m+1.96*se):+.2f}), median {100*st.median(rel):+.2f}%, n={len(rel)}")
