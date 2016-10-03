%%
clear all, close all; clc;
N=100  ; % image size
M=6  ;% number faces

scrsz = get(0,'Screensize');
ancho = scrsz(3);
alto = scrsz(4);
ancho = round(ancho/3);
alto = round(alto/2)-40;


%% matheus1
MC.names = {'matheus1','matheus2','matheus3','matheus4','matheus5','matheus6'};
% MC = zeros(N,N,M);
% MC = uint8(MC);

for h=1:6
    matheus = imread(['matheus\matheus',num2str(h),'.JPG'],'jpg');
    matheus=rgb2gray(matheus);
    matheus=imresize(matheus,[N N] );
    MC.data{h} = matheus;
end
%
figure('Position',[10 alto ancho alto])
subplot(2,3,1),imshow(MC.data{1},'Initialmagnification','fit');;title(MC.names{1})


% alex1
SH.names = {'alex1','alex2','alex3','alex4','alex5','alex6'};

% SH = zeros(N,N,M);
% SH = uint8(SH);
for h=1:6
    alex = imread(['alex\alex',num2str(h),'.JPG'],'jpg');
    alex=rgb2gray(alex);
    alex=imresize(alex,[N N] );
    SH.data{h} = alex;
end
subplot(2,3,2),imshow(SH.data{1},'Initialmagnification','fit');;title(SH.names{1})

% bruno1
LN.names = {'bruno1','bruno2','bruno3','bruno4','bruno5','bruno6'};
% LN = zeros(N,N,M);
% LN = uint8(LN);
for h=1:6
    bruno = imread(['bruno\bruno',num2str(h),'.JPG'],'jpg');
    bruno=rgb2gray(bruno);
    bruno=imresize(bruno,[N N] );
    LN.data{h} = bruno;
end
subplot(2,3,3),imshow(LN.data{1},'Initialmagnification','fit');;title(LN.names{1})


% filipe1
LB.names = {'filipe1','filipe2','filipe3','filipe4','filipe5','filipe6'};
% LB = zeros(N,N,M);
% LB = uint8(LB);
for h=1:6
    filipe = imread(['filipe\filipe',num2str(h),'.JPG'],'jpg');
    filipe=rgb2gray(filipe);
    filipe=imresize(filipe,[N N] );
    LB.data{h} = filipe;
end

subplot(2,3,4),imshow(LB.data{1},'Initialmagnification','fit');;title(LB.names{1})

% maria
EX.names = {'maria1','maria1','maria1','maria1','maria1','maria1'};
% EX = zeros(N,N,M);
% EX = uint8(EX);
for h=1:6
    maria = imread(['maria\maria',num2str(h),'.JPG'],'jpg');
    maria=rgb2gray(maria);
    maria=imresize(maria,[N N] );
    EX.data{h} = maria;
end
subplot(2,3,5),imshow(EX.data{1},'Initialmagnification','fit');;title(EX.names{1})

% rapha
GR.names = {'rapha.1','rapha.2','rapha.3','rapha.4','rapha.5','rapha.6'}
% GR = zeros(N,N,M);
% GR = uint8(GR);
for h=1:6
    rapha = imread(['rapha\rapha',num2str(h),'.JPG'],'jpg');
    rapha=rgb2gray(rapha);
    rapha=imresize(rapha,[N N] );
    GR.data{h} = rapha;
end
subplot(2,3,6),imshow(GR.data{1},'Initialmagnification','fit');;title(GR.names{1})

pause(1)
%% store



st.names = {'matheus','alex','bruno','filipe','maria1','rapha'};
st.data{1} = MC;
st.data{2} = SH;
st.data{3} = LN;
st.data{4} = LB;
st.data{5} = EX;
st.data{6} = GR;
%%
% z  = [matheus1  alex1  bruno1 ;     filipe1  maria1 rapha];
% figure(6),imshow(z,'Initialmagnification','fit');;title('z')



save classFile st;



% clear all, close all;
load classFile;
M=6;N=100;
avImg=zeros(N);
for f=1:6
    AVI.data{f} = im2single(avImg);
end

% z  = [ st.data{1}  st.data{2}    st.data{3}; st.data{4}     st.data{5}  st.data{6}];

%% compute mean
figure('Position',[ancho alto ancho alto])
for l=1:M
    for k=1:M
        st.data{k}.data{l} = im2single(st.data{k}.data{l});
        AVI.data{l}   =AVI.data{l}  + (1/M)*st.data{k}.data{l};
        if l==1
            subplot(2,3,k),imshow(AVI.data{l},'Initialmagnification','fit');title('average')
            pause(1)
        end
    end
end
pause(1)


%% normalize (remove mean)
for l=1:M
    for k=1:M    
    st.dataAvg{k}.dataAvg{l}  = st.data{k}.data{l} -AVI.data{l};
    end
end

z  = [ st.dataAvg{1}.dataAvg{1}  st.dataAvg{2}.dataAvg{1}   st.dataAvg{5}.dataAvg{1}  ; 
       st.dataAvg{3}.dataAvg{1}  st.dataAvg{4}.dataAvg{1}   st.dataAvg{6}.dataAvg{1}];


% ZC(:,:,1) = st.dataAvg{1}(:,:,1);
% ZC(:,:,2) = st.dataAvg{2}(:,:,1);
% ZC(:,:,3) = st.dataAvg{3}(:,:,1);
% ZC(:,:,4) = st.dataAvg{4}(:,:,1);
% ZC(:,:,5) = st.dataAvg{5}(:,:,1);
% ZC(:,:,6) = st.dataAvg{6}(:,:,1);

figure('Position',[(2*ancho) alto ancho alto]),imshow(z,'Initialmagnification','fit');;title('z average')
pause(1)

%% generate A = [ img1(:)  img2(:) ...  imgM(:) ];
A = zeros(N*N,M);% (N*N)*M   10000*6
for p=1:M
    Ast.data{p} =A; 
end
for k=1:M
    for l=1:M
    Ast.data{k}(:,l) = st.dataAvg{l}.dataAvg{k}(:);
    end
end
% covariance matrix small dimension (transposed)
C = Ast.data{1}'*Ast.data{1};
C2 = Ast.data{2}'*Ast.data{2};
C3 = Ast.data{3}'*Ast.data{3};
C4 = Ast.data{4}'*Ast.data{4};
C5 = Ast.data{5}'*Ast.data{5};
C6 = Ast.data{6}'*Ast.data{6};

% figure(4),imagesc(C);title('covariance')

%% eigen vectros  in small dimension
[   Veigvec,Deigval ]  = eig(C);% v M*M e M*M only diagonal 4 eigen values
[   Veigvec2,Deigval2 ]  = eig(C2);
[   Veigvec3,Deigval3 ]  = eig(C3);
[   Veigvec4,Deigval4 ]  = eig(C4);
[   Veigvec5,Deigval5 ]  = eig(C5);
[   Veigvec6,Deigval6 ]  = eig(C6);
% eigan face in large dimension  A*veigvec is eigen vector of Clarge
Vlarge = Ast.data{1}*Veigvec;% 2500*M*M*M  =2500 *M
Vlarge2 = Ast.data{2}*Veigvec;
Vlarge3 = Ast.data{3}*Veigvec;
Vlarge4 = Ast.data{4}*Veigvec;
Vlarge5 = Ast.data{5}*Veigvec;
Vlarge6 = Ast.data{6}*Veigvec;

% reshape to eigen face
eigenfaces=[];
eigenfaces2=[];
eigenfaces3=[];
eigenfaces4=[];
eigenfaces5=[];
eigenfaces6=[];

for k=1:M
    c  = Vlarge(:,k);
    eigenfaces{k} = reshape(c,N,N);
end

for k=1:M
    c2  = Vlarge2(:,k);
    eigenfaces2{k} = reshape(c2,N,N);
end

for k=1:M
    c3  = Vlarge3(:,k);
    eigenfaces3{k} = reshape(c3,N,N);
end

for k=1:M
    c4  = Vlarge4(:,k);
    eigenfaces4{k} = reshape(c4,N,N);
end

for k=1:M
    c5  = Vlarge5(:,k);
    eigenfaces5{k} = reshape(c5,N,N);
end

for k=1:M
    c6  = Vlarge6(:,k);
    eigenfaces6{k} = reshape(c6,N,N);
end


x=diag(Deigval);
x2=diag(Deigval2);
x3=diag(Deigval3);
x4=diag(Deigval4);
x5=diag(Deigval5);
x6=diag(Deigval6);


[xc,xci]=sort(x,'descend');% largest eigenval
[xc2,xci2]=sort(x2,'descend');
[xc3,xci3]=sort(x3,'descend');
[xc4,xci4]=sort(x4,'descend');
[xc5,xci5]=sort(x5,'descend');
[xc6,xci6]=sort(x6,'descend');

z  = [ eigenfaces{xci(1)}  eigenfaces{xci(2)}   eigenfaces{xci(3)} ;
    eigenfaces{xci(4)}     eigenfaces{xci(5)}   eigenfaces{xci(6)}];
figure('Position',[10 10 ancho alto]),imshow(z,'Initialmagnification','fit');;title('eigenfaces')
pause(2)

%% weights
nsel=6% select  eigen faces
for mi=1:M  % image number
    for k=1:nsel   % eigen face for coeff number
        wi(mi,k) =   sum(Ast.data{1}(:,mi).* eigenfaces{xci(k)}(:)) ;
    end
end

for mi=1:M  % image number
    for k=1:nsel   % eigen face for coeff number
        wi2(mi,k) =   sum(Ast.data{2}(:,mi).* eigenfaces{xci2(k)}(:)) ;
    end
end

for mi=1:M  % image number
    for k=1:nsel   % eigen face for coeff number
        wi3(mi,k) =   sum(Ast.data{3}(:,mi).* eigenfaces{xci3(k)}(:)) ;
    end
end

for mi=1:M  % image number
    for k=1:nsel   % eigen face for coeff number
        wi4(mi,k) =   sum(Ast.data{4}(:,mi).* eigenfaces{xci4(k)}(:)) ;
    end
end

for mi=1:M  % image number
    for k=1:nsel   % eigen face for coeff number
        wi5(mi,k) =   sum(Ast.data{5}(:,mi).* eigenfaces{xci5(k)}(:)) ;
    end
end

for mi=1:M  % image number
    for k=1:nsel   % eigen face for coeff number
        wi6(mi,k) =   sum(Ast.data{6}(:,mi).* eigenfaces{xci6(k)}(:)) ;
    end
end



%% classify new img  matheus
% folder work C:\Users\michaels.DSI\Desktop\faces\class\
testFaceMic = imread('teste.jpg','jpg');
testFaceMic  =rgb2gray(testFaceMic);
testFaceMic = imresize(testFaceMic,[N N]);
testFaceMic   =  im2single(testFaceMic);
% testFaceMic =  st.data{1}; %test


figure('Position',[ancho 10 ancho alto]), imshow(testFaceMic,'Initialmagnification','fit'); title('test face')
pause(2)
Aface = testFaceMic(:)-AVI.data{1}(:); % normilized face
Aface2 = testFaceMic(:)-AVI.data{2}(:);
Aface3 = testFaceMic(:)-AVI.data{3}(:);
Aface4 = testFaceMic(:)-AVI.data{4}(:);
Aface5 = testFaceMic(:)-AVI.data{5}(:);
Aface6 = testFaceMic(:)-AVI.data{6}(:);

for(tt=1:nsel)
    wface(tt)  =  sum(Aface.* eigenfaces{xci(tt)}(:)) ;
end

for(tt=1:nsel)
    wface2(tt)  =  sum(Aface2.* eigenfaces{xci2(tt)}(:)) ;
end

for(tt=1:nsel)
    wface3(tt)  =  sum(Aface3.* eigenfaces{xci3(tt)}(:)) ;
end

for(tt=1:nsel)
    wface4(tt)  =  sum(Aface4.* eigenfaces{xci4(tt)}(:)) ;
end

for(tt=1:nsel)
    wface5(tt)  =  sum(Aface5.* eigenfaces{xci5(tt)}(:)) ;
end

for(tt=1:nsel)
    wface6(tt)  =  sum(Aface6.* eigenfaces{xci6(tt)}(:)) ;
end

%% compute distance
for mi=1:M
    fsumcur=0;
    for(tt=1:nsel)
        fsumcur = fsumcur + (wface(tt) -wi(mi,tt)).^2;
    end
    diffWeights.data{1}(mi) =   sqrt( fsumcur);
end

for mi=1:M
    fsumcur2=0;
    for(tt=1:nsel)
        fsumcur2 = fsumcur2 + (wface2(tt) -wi2(mi,tt)).^2;
    end
    diffWeights.data{2}(mi) =   sqrt( fsumcur2);
end

for mi=1:M
    fsumcur3=0;
    for(tt=1:nsel)
        fsumcur3 = fsumcur3 + (wface3(tt) -wi3(mi,tt)).^2;
    end
    diffWeights.data{3}(mi) =   sqrt( fsumcur3);
end

for mi=1:M
    fsumcur4=0;
    for(tt=1:nsel)
        fsumcur4 = fsumcur4 + (wface4(tt) -wi4(mi,tt)).^2;
    end
    diffWeights.data{4}(mi) =   sqrt( fsumcur4);
end

for mi=1:M
    fsumcur5=0;
    for(tt=1:nsel)
        fsumcur5 = fsumcur5 + (wface5(tt) -wi5(mi,tt)).^2;
    end
    diffWeights.data{5}(mi) =   sqrt( fsumcur5);
end

for mi=1:M
    fsumcur6=0;
    for(tt=1:nsel)
        fsumcur6 = fsumcur6 + (wface6(tt) -wi6(mi,tt)).^2;
    end
    diffWeights.data{6}(mi) =   sqrt( fsumcur6);
end


% mic classified as 5 ..

[val.data{1} in.data{1}]= min(diffWeights.data{1});
[val.data{2} in.data{2}]= min(diffWeights.data{2});
[val.data{3} in.data{3}]= min(diffWeights.data{3});
[val.data{4} in.data{4}]= min(diffWeights.data{4});
[val.data{5} in.data{5}]= min(diffWeights.data{5});
[val.data{6} in.data{6}]= min(diffWeights.data{6});




figure('Position',[2*ancho 10 ancho alto]), imshow(st.data{in.data{1}}.data{1}), title(['The image corresponds to ', st.names{in.data{1}}])
figure('Position',[2*ancho 10 ancho alto]), imshow(st.data{in.data{2}}.data{2}), title(['The image corresponds to ', st.names{in.data{2}}])
figure('Position',[2*ancho 10 ancho alto]), imshow(st.data{in.data{3}}.data{3}), title(['The image corresponds to ', st.names{in.data{3}}])
figure('Position',[2*ancho 10 ancho alto]), imshow(st.data{in.data{4}}.data{4}), title(['The image corresponds to ', st.names{in.data{4}}])
figure('Position',[2*ancho 10 ancho alto]), imshow(st.data{in.data{5}}.data{5}), title(['The image corresponds to ', st.names{in.data{5}}])
figure('Position',[2*ancho 10 ancho alto]), imshow(st.data{in.data{6}}.data{6}), title(['The image corresponds to ', st.names{in.data{6}}])



% %% classify new img  bruno
% testFacebruno = imread('100_2120.jpg','jpg');
% testFacebruno  =rgb2gray(testFacebruno);
% testFacebruno = imresize(testFacebruno,[N N]);
% testFacebruno   =  im2single(testFacebruno);
% figure(7), imshow(testFacebruno,'Initialmagnification','fit'); title('test face bruno')
% Aface = testFacebruno(:)-AVI.data{1}(:);
% for(tt=1:nsel)
%   wface(tt)  =  sum(Aface.* eigenfaces{xci(tt)}(:)) ;
% end
%
%
% % compute distance
% for mi=1:M
%     fsumcur=0;
%     for(tt=1:nsel)
%         fsumcur = fsumcur + (wface(tt) -wi(mi,tt)).^2;
%     end
%     diffWeights(mi) =   sqrt( fsumcur);
% end
% % bruno classified as filipe
%
%
% %% filipe3.jpg
% testFacefilipe = imread('filipe3.jpg','jpg');
% testFacefilipe  =rgb2gray(testFacefilipe);
% testFacefilipe = imresize(testFacefilipe,[N N]);
% testFacefilipe   =  im2single(testFacefilipe);
% figure(8), imshow(testFacefilipe,'Initialmagnification','fit'); title('test face testFacefilipe')
% Aface = testFacefilipe(:)-AVI.data{1}(:);
% wface=[];
% for(tt=1:nsel)
%   wface(tt)  =  sum(Aface.* eigenfaces{xci(tt)}(:)) ;
% end
%
%
% % compute distance
% for mi=1:M
%     fsumcur=0;
%     for(tt=1:nsel)
%         fsumcur = fsumcur + (wface(tt) -wi(mi,tt)).^2;
%     end
%     diffWeights(mi) =   sqrt( fsumcur);
% end
% diffWeights  =diffWeights.';