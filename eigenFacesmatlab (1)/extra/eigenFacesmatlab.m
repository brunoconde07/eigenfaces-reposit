%%
clear all, close all; clc;
N=100  ; % image size
M=6  ;% number faces

scrsz = get(0,'Screensize');
ancho = scrsz(3);
alto = scrsz(4);
ancho = round(ancho/3);
alto = round(alto/2)-40;


%% mic1
MC = zeros(N,N,M);
MC = uint8(MC);
for h=1:6
    mic1 = imread(['mic\mic',num2str(h),'.JPG'],'jpg');
    mic1=rgb2gray(mic1);
    mic1=imresize(mic1,[N N] );
    MC(:,:,h) = mic1;
end
%
figure('Position',[10 alto ancho alto])
subplot(2,3,1),imshow(MC(:,:,1),'Initialmagnification','fit');;title('mic1')


% shelly1
SH = zeros(N,N,M);
SH = uint8(SH);
for h=1:6
    shelly1 = imread(['shelly\shelly',num2str(h),'.JPG'],'jpg');
    shelly1=rgb2gray(shelly1);
    shelly1=imresize(shelly1,[N N] );
    SH(:,:,h) = shelly1;
end
subplot(2,3,2),imshow(SH(:,:,1),'Initialmagnification','fit');;title('shelly1')

% linoy1
LN = zeros(N,N,M);
LN = uint8(LN);
for h=1:6
    linoy1 = imread(['linoy\linoy',num2str(h),'.JPG'],'jpg');
    linoy1=rgb2gray(linoy1);
    linoy1=imresize(linoy1,[N N] );
    LN(:,:,h) = linoy1;
end
subplot(2,3,3),imshow(LN(:,:,1),'Initialmagnification','fit');;title('linoy1')


% libi1
LB = zeros(N,N,M);
LB = uint8(LB);
for h=1:6
    libi1 = imread(['libi\libi',num2str(h),'.JPG'],'jpg');
    libi1=rgb2gray(libi1);
    libi1=imresize(libi1,[N N] );
    LB(:,:,h) = libi1;
end

subplot(2,3,4),imshow(LB(:,:,1),'Initialmagnification','fit');;title('libi1')

% extra face
EX = zeros(N,N,M);
EX = uint8(EX);
for h=1:6
    extraface1 = imread(['extra\extraface',num2str(h),'.JPG'],'jpg');
    extraface1=rgb2gray(extraface1);
    extraface1=imresize(extraface1,[N N] );
    EX(:,:,h) = extraface1;
end
subplot(2,3,5),imshow(EX(:,:,1),'Initialmagnification','fit');;title('extraface1')

% m_gr.jpg
GR = zeros(N,N,M);
GR = uint8(GR);
for h=1:6
    extraface2 = imread(['gr\gr',num2str(h),'.JPG'],'jpg');
    extraface2=rgb2gray(extraface2);
    extraface2=imresize(extraface2,[N N] );
    GR(:,:,h) = extraface2;
end
subplot(2,3,6),imshow(GR(:,:,1),'Initialmagnification','fit');;title('extraface2')

pause(1)
%% store



st.names = {'mic','shelly','linoy','libi','exface1','exface2'};
st.data{1} = MC;
st.data{2} = SH;
st.data{3} = LN;
st.data{4} = LB;
st.data{5} = EX;
st.data{6} = GR;
%%
% z  = [mic1  shelly1  linoy1 ;     libi1  extraface1 extraface2];
% figure(6),imshow(z,'Initialmagnification','fit');;title('z')



save classFile st;



% clear all, close all;
load classFile;
M=6;N=100;
avImg=zeros(N,N,M);

% z  = [ st.data{1}  st.data{2}    st.data{3}; st.data{4}     st.data{5}  st.data{6}];
figure('Position',[ancho alto ancho alto])
%% compute mean
for k=1:M
    for l=1:M
        %     st.data{l}(:,:,k) = im2single((st.data{l}(:,:,k)));
        avImg(:,:,k)=avImg(:,:,k)  + (1/M)*(im2single((st.data{l}(:,:,k))));
        if k==1
            subplot(2,3,l),imshow(avImg(:,:,k),'Initialmagnification','fit');title('average')
            pause(1)
        end
    end
    
end
pause(2)


%% normalize (remove mean)
for k=1:M
    for l=1:M
    st.dataAvg{k}(:,:,l)  = double(st.data{k}(:,:,l)) -avImg(:,:,l);
    end
end

z  = [ st.dataAvg{1}(:,:,1)  st.dataAvg{2}(:,:,1)   st.dataAvg{5}(:,:,1)  ; 
    st.dataAvg{3}(:,:,1)     st.dataAvg{4}(:,:,1)  st.dataAvg{6}(:,:,1)];

z2  = [ st.dataAvg{1}(:,:,2)  st.dataAvg{2}(:,:,2)   st.dataAvg{5}(:,:,2)  ; 
    st.dataAvg{3}(:,:,2)     st.dataAvg{4}(:,:,2)  st.dataAvg{6}(:,:,2)];
z3  = [ st.dataAvg{1}(:,:,3)  st.dataAvg{2}(:,:,3)   st.dataAvg{5}(:,:,3)  ; 
    st.dataAvg{3}(:,:,3)     st.dataAvg{4}(:,:,3)  st.dataAvg{6}(:,:,3)];
z4  = [ st.dataAvg{1}(:,:,4)  st.dataAvg{2}(:,:,4)   st.dataAvg{5}(:,:,4)  ; 
    st.dataAvg{3}(:,:,4)     st.dataAvg{4}(:,:,4)  st.dataAvg{6}(:,:,4)];
z5  = [ st.dataAvg{1}(:,:,5)  st.dataAvg{2}(:,:,5)   st.dataAvg{5}(:,:,5)  ; 
    st.dataAvg{3}(:,:,5)     st.dataAvg{4}(:,:,5)  st.dataAvg{6}(:,:,5)];
z6  = [ st.dataAvg{1}(:,:,6)  st.dataAvg{2}(:,:,6)   st.dataAvg{5}(:,:,6)  ; 
    st.dataAvg{3}(:,:,6)     st.dataAvg{4}(:,:,6)  st.dataAvg{6}(:,:,6)];


ZC(:,:,1) = st.dataAvg{1}(:,:,1);
ZC(:,:,2) = st.dataAvg{2}(:,:,1);
ZC(:,:,3) = st.dataAvg{3}(:,:,1);
ZC(:,:,4) = st.dataAvg{4}(:,:,1);
ZC(:,:,5) = st.dataAvg{5}(:,:,1);
ZC(:,:,6) = st.dataAvg{6}(:,:,1);
figure('Position',[(2*ancho) alto ancho alto]),imshow(z,'Initialmagnification','fit');;title('z average')
pause(2)

%% generate A = [ img1(:)  img2(:) ...  imgM(:) ];
A = zeros(N*N,M,M);% (N*N)*M   2500*4
for k=1:M
    for l=1:M
    A(:,l,k) = st.dataAvg{l}(:,:,k);
    end
end
% covariance matrix small dimension (transposed)
C = A(:,:,1)'*A(:,:,1);
C2 = A(:,:,2)'*A(:,:,2);
C3 = A(:,:,3)'*A(:,:,3);
C4 = A(:,:,4)'*A(:,:,4);
C5 = A(:,:,5)'*A(:,:,5);
C6 = A(:,:,6)'*A(:,:,6);
% figure(4),imagesc(C);title('covariance')

%% eigen vectros  in small dimension
[   Veigvec,Deigval ]  = eig(C);% v M*M e M*M only diagonal 4 eigen values
% eigan face in large dimension  A*veigvec is eigen vector of Clarge
Vlarge = A*Veigvec;% 2500*M*M*M  =2500 *M
% reshape to eigen face
eigenfaces=[];
for k=1:M
    c  = Vlarge(:,k);
    eigenfaces{k} = reshape(c,N,N);
end
x=diag(Deigval);
[xc,xci]=sort(x,'descend');% largest eigenval
z  = [ eigenfaces{xci(1)}  eigenfaces{xci(2)}   eigenfaces{xci(3)} ;
    eigenfaces{xci(4)}     eigenfaces{xci(5)}   eigenfaces{xci(6)}];
figure('Position',[10 10 ancho alto]),imshow(z,'Initialmagnification','fit');;title('eigenfaces')
pause(2)

%% weights
nsel=6% select  eigen faces
for mi=1:M  % image number
    for k=1:nsel   % eigen face for coeff number
        wi(mi,k) =   sum(A(:,mi).* eigenfaces{xci(k)}(:)) ;
    end
end

%% classify new img  mic
% folder work C:\Users\michaels.DSI\Desktop\faces\class\
testFaceMic = imread('teste\teste.jpg','jpg');
testFaceMic  =rgb2gray(testFaceMic);
testFaceMic = imresize(testFaceMic,[N N]);
testFaceMic   =  im2single(testFaceMic);
% testFaceMic =  st.data{1}; %test


figure('Position',[ancho 10 ancho alto]), imshow(testFaceMic,'Initialmagnification','fit'); title('test face michael')
pause(2)
Aface = testFaceMic(:)-avImg(:); % normilized face


for(tt=1:nsel)
    wface(tt)  =  sum(Aface.* eigenfaces{xci(tt)}(:)) ;
end


%% compute distance
for mi=1:M
    fsumcur=0;
    for(tt=1:nsel)
        fsumcur = fsumcur + (wface(tt) -wi(mi,tt)).^2;
    end
    diffWeights(mi) =   sqrt( fsumcur);
end
% mic classified as 5 ..
[val in]= min(diffWeights);
figure('Position',[2*ancho 10 ancho alto]), imshow(st.data{in}), title(['The image corresponds to ', st.names{in}])

% %% classify new img  linoy
% testFaceLinoy = imread('100_2120.jpg','jpg');
% testFaceLinoy  =rgb2gray(testFaceLinoy);
% testFaceLinoy = imresize(testFaceLinoy,[N N]);
% testFaceLinoy   =  im2single(testFaceLinoy);
% figure(7), imshow(testFaceLinoy,'Initialmagnification','fit'); title('test face linoy')
% Aface = testFaceLinoy(:)-avImg(:);
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
% % linoy classified as libi
%
%
% %% libi3.jpg
% testFaceLibi = imread('libi3.jpg','jpg');
% testFaceLibi  =rgb2gray(testFaceLibi);
% testFaceLibi = imresize(testFaceLibi,[N N]);
% testFaceLibi   =  im2single(testFaceLibi);
% figure(8), imshow(testFaceLibi,'Initialmagnification','fit'); title('test face testFaceLibi')
% Aface = testFaceLibi(:)-avImg(:);
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