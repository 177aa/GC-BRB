function F = gg_features(e)
%GG_FEATURES Preserve the original 37-column definition and order.
weighted=@(x,w)sum(x.*w,2)./max(sum(w,2),eps);
wr=weighted(e.diffusion,e.confidence);
wh1=weighted(e.onehop,e.confidence);
wh2=weighted(e.twohop,e.confidence);
neutral=sum(e.confidence,2)<=eps;
wr(neutral)=0.5;wh1(neutral)=0.5;wh2(neutral)=0.5;
coverage=mean(e.presence,2);
sr=zeros(size(coverage));sh1=sr;sh2=sr;
for i=1:numel(coverage)
    use=e.presence(i,:);
    if nnz(use)>=2
        sr(i)=std(e.diffusion(i,use),0,2);
        sh1(i)=std(e.onehop(i,use),0,2);
        sh2(i)=std(e.twohop(i,use),0,2);
    end
end
F=[e.omics_score,e.diffusion,e.onehop,e.twohop,e.confidence, ...
    double(e.presence),wr,wh1,wh2,coverage,sr,0.5*(sh1+sh2)];
if any(~isfinite(F(:))), error('GG:Features','Nonfinite evidence.'); end
end
