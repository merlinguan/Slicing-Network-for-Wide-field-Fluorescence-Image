clear,clc
load('net_2d.mat','net_2d');


for i = 1:50
imdsTest1 = imageDatastore(sprintf(".../raw/%03d.tiff",i)); % 离焦图像
A = imread(sprintf(".../raw/%03d.tiff",i)); % 离焦图像
Y1 =  predict(net_2d,imdsTest1);
B = uint8(Y1); 
C = imread(sprintf(".../gt/%03d.tiff",i));
imwrite(uint8(Y1),sprintf(".../net/%03d.tiff",i));

montage({double(A),double(B),double(C)},'DisplayRange',[0,255],'Size',[1,3],...
        'BorderSize',[2,2]);
    colormap parula
    drawnow
    pause
end
