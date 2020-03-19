% load data
load mnist-original.mat
train_image = data(1:784,1:60000);
test_image = data(1:784, 60001:70000);
train_label = label(1,1:60000);
test_label = label(1,60001:70000);

%read digits
test1 = train_image(1:784,1);
read_digit(test1);
test2 = train_image(1:784,20000);
read_digit(test2);
test3 = train_image(1:784,40000);
read_digit(test3);
test4 = train_image(1:784,50000);
read_digit(test4);

%average digits
digit0 = mean(train_image(1:784,1:5923),2);
read_digit(digit0)
digit1 = mean(train_image(1:784,5924:12665),2);
read_digit(digit1)
digit2 = mean(train_image(1:784,12666:18623),2);
read_digit(digit2)
digit3 = mean(train_image(1:784,18624:24754),2);
read_digit(digit3)
digit4 = mean(train_image(1:784,24755:30596),2);
read_digit(digit4)
digit5 = mean(train_image(1:784,30597:36017),2);
read_digit(digit5)
digit6 = mean(train_image(1:784,36018:41935),2);
read_digit(digit6)
digit7 = mean(train_image(1:784,41936:48200),2);
read_digit(digit7)
digit8 = mean(train_image(1:784,48201:54051),2);
read_digit(digit8)
digit9 = mean(train_image(1:784,54052:60000),2);
read_digit(digit9)

%neuron test
images = double(train_image(1:784,1));
weight = -1 + 2*rand(784,1);
out = neuron(images,weight);
disp(out); %output:1

%initialize network
in = double(train_image(1:784,1));
layers = 5;
[WM,output] = build_network(in,layers);
disp(output);

%read digits funtion
function read_digit(data)
      data = reshape(data,[28,28]);
      image(rot90(flipud(data),-1)),colormap(gray(256));
end

%neuron function
function out = neuron (O,W)
  n = length(O);
  net = 0;
  for i = 1:n
      net = net + O(i)*W(i);
  end
  out = 1 / (1 + exp (-net));
end

%network function
function [W,output] = build_network(inputs,layers)
    input = reshape(inputs,[28,28]); %read pixels into input layer
    layer_1 = [];
        layer_1 = [layer_1 input(1,:)];
        layer_1 = [layer_1 input(2:28,28).'];
        layer_1 = [layer_1 input(28,flip(1:27))];
        layer_1 = [layer_1 input(flip(3:27),1).'];
     for i=1:13
        layer_1 = [layer_1 input(i+1,i:28-i)];
        layer_1 = [layer_1 input(i+2:28-i,28-i).']; 
        layer_1 = [layer_1 input(28-i,flip(i+1:27-i))];
        layer_1 = [layer_1 input(flip(i+3:27-i), i+1).'];
     end
     W = {zeros(784,512)}; %store weight matrices
     for j = 1:3
         W{j+1} = zeros(512/ (2^(j-1)),256/ (2^(j-1)));
     end
     for n = 1:512 %first hidden layer :512 nodes
         weights = -1 + 2*rand(784,1);
         W{1,1}(:,n) = weights;
     end
     net =transpose(W{1,1})*transpose(layer_1);
     layer_2 =1./(1+exp(-net));
     for n = 1:256 %second hidden layer :256 nodes
         weights = -1 + 2*rand(512,1);
         W{1,2}(:,n) = weights;
     end
     net =transpose(W{1,2})*layer_2;
     layer_3 =1./(1+exp(-net));
     for n = 1:128 %third hidden layer :128 nodes
         weights = -1 + 2*rand(256,1);
         W{1,3}(:,n) = weights;
     end
     net =transpose(W{1,3})*layer_3;
     layer_4 =1./(1+exp(-net));
     for n = 1:64 %fourth hidden layer :64 nodes
         weights = -1 + 2*rand(128,1); 
         W{1,4}(:,n) = weights;
     end
     net =transpose(W{1,4})*layer_4;
     layer_5 =1./(1+exp(-net));
     W{5} = zeros(64,10); %get output layer: 10 nodes
     for n = 1:10
         weights = -1 + 2*rand(64,1);
         W{1,5}(:,n) = weights;    
     end
     net = transpose(W{1,5})*layer_5;
     output = 1./(1+exp(-net));    
end
