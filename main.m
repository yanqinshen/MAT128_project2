format long

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
%set weights
W = {zeros(784,512)}; %store weight matrices
for j = 1:3
    W{j+1} = zeros(512/ (2^(j-1)),256/ (2^(j-1)));
end
for n = 1:512
    weights = -1 + 2*rand(784,1);
    W{1,1}(:,n) = weights;
end
for n = 1:256
    weights = -1 + 2*rand(512,1);
    W{1,2}(:,n) = weights;
end
for n = 1:128
    weights = -1 + 2*rand(256,1);
    W{1,3}(:,n) = weights;
end
for n = 1:64
    weights = -1 + 2*rand(128,1);
    W{1,4}(:,n) = weights;
end
for n = 1:10
    weights = -1 + 2*rand(64,1);
    W{1,5}(:,n) = weights;
end
layers = 4
[layers_val,output] = build_network(in,W,layers);
disp(output);

% train network test
digit = double(train_label(1,1));
weight = train_network(layers_val,digit,output,W,layers);

%parameters

%3 layers
W = {zeros(784,512)}; %store weight matrices
for j = 1:2
    W{j+1} = zeros(512/ (2^(j-1)),256/ (2^(j-1)));
end
for n = 1:512
    weights = -1 + 2*rand(784,1);
    W{1,1}(:,n) = weights;
end
for n = 1:256
    weights = -1 + 2*rand(512,1);
    W{1,2}(:,n) = weights;
end
for n = 1:128
    weights = -1 + 2*rand(256,1);
    W{1,3}(:,n) = weights;
end
for n = 1:10
    weights = -1 + 2*rand(128,1);
    W{1,4}(:,n) = weights;
end
layers = 3;
n = 60000;
[rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label);
disp(rate1);
disp(rate2);

%4 layers
W = {zeros(784,512)}; %store weight matrices
for j = 1:3
    W{j+1} = zeros(512/ (2^(j-1)),256/ (2^(j-1)));
end
for n = 1:512
    weights = -1 + 2*rand(784,1);
    W{1,1}(:,n) = weights;
end
for n = 1:256
    weights = -1 + 2*rand(512,1);
    W{1,2}(:,n) = weights;
end
for n = 1:128
    weights = -1 + 2*rand(256,1);
    W{1,3}(:,n) = weights;
end
for n = 1:64
    weights = -1 + 2*rand(128,1);
    W{1,4}(:,n) = weights;
end
for n = 1:10
    weights = -1 + 2*rand(64,1);
    W{1,5}(:,n) = weights;
end
layers = 4;
n = 60000;
[rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label);
disp(rate1);
disp(rate2);

%5 layers
W = {zeros(784,512)}; %store weight matrices
for j = 1:4
    W{j+1} = zeros(512/ (2^(j-1)),256/ (2^(j-1)));
end
for n = 1:512
    weights = -1 + 2*rand(784,1);
    W{1,1}(:,n) = weights;
end
for n = 1:256
    weights = -1 + 2*rand(512,1);
    W{1,2}(:,n) = weights;
end
for n = 1:128
    weights = -1 + 2*rand(256,1);
    W{1,3}(:,n) = weights;
end
for n = 1:64
    weights = -1 + 2*rand(128,1);
    W{1,4}(:,n) = weights;
end
for n = 1:32
    weights = -1 + 2*rand(64,1);
    W{1,5}(:,n) = weights;
end
for n = 1:10
    weights = -1 + 2*rand(32,1);
    W{1,6}(:,n) = weights;
end
layers = 5;
n = 60000;
[rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label);
disp(rate1);
disp(rate2);

%500,400,300,200,100
W = {zeros(784,500)}; %store weight matrices
for n = 1:500
    weights = -1 + 2*rand(784,1);
    W{1,1}(:,n) = weights;
end
for n = 1:400
    weights = -1 + 2*rand(500,1);
    W{1,2}(:,n) = weights;
end
for n = 1:300
    weights = -1 + 2*rand(400,1);
    W{1,3}(:,n) = weights;
end
for n = 1:200
    weights = -1 + 2*rand(300,1);
    W{1,4}(:,n) = weights;
end
for n = 1:100
    weights = -1 + 2*rand(200,1);
    W{1,5}(:,n) = weights;
end
for n = 1:10
    weights = -1 + 2*rand(100,1);
    W{1,6}(:,n) = weights;
end
layers = 5;
n = 60000;
[rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label);
disp(rate1);
disp(rate2);

%300,200,100,50,25
W = {zeros(784,300)}; %store weight matrices
for n = 1:300
    weights = -1 + 2*rand(784,1);
    W{1,1}(:,n) = weights;
end
for n = 1:200
    weights = -1 + 2*rand(300,1);
    W{1,2}(:,n) = weights;
end
for n = 1:100
    weights = -1 + 2*rand(200,1);
    W{1,3}(:,n) = weights;
end
for n = 1:50
    weights = -1 + 2*rand(100,1);
    W{1,4}(:,n) = weights;
end
for n = 1:25
    weights = -1 + 2*rand(50,1);
    W{1,5}(:,n) = weights;
end
for n = 1:10
    weights = -1 + 2*rand(25,1);
    W{1,6}(:,n) = weights;
end
layers = 5;
n = 60000;
[rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label);
disp(rate1);
disp(rate2);

%n = 50000
W = {zeros(784,500)}; %store weight matrices
for n = 1:500
    weights = -1 + 2*rand(784,1);
    W{1,1}(:,n) = weights;
end
for n = 1:400
    weights = -1 + 2*rand(500,1);
    W{1,2}(:,n) = weights;
end
for n = 1:300
    weights = -1 + 2*rand(400,1);
    W{1,3}(:,n) = weights;
end
for n = 1:200
    weights = -1 + 2*rand(300,1);
    W{1,4}(:,n) = weights;
end
for n = 1:100
    weights = -1 + 2*rand(200,1);
    W{1,5}(:,n) = weights;
end
for n = 1:10
    weights = -1 + 2*rand(100,1);
    W{1,6}(:,n) = weights;
end
layers = 5;
n = 50000;
[rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label);
disp(rate1);
disp(rate2);

%40000
n = 40000;
[rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label);
disp(rate1);
disp(rate2);

%30000
n = 30000;
[rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label);
disp(rate1);
disp(rate2);

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
function [layers_val,output] = build_network(inputs,W,layers)
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
     layers_val{1} = layer_1.';
     net =transpose(W{1,1})*transpose(layer_1); %first hidden layer: 512 nodes
     layer =1./(1+exp(-net));
     layers_val{2}  = layer;
     for i = 2:layers
        net =transpose(W{1,i})*layer; %second hidden layer :256 nodes
        layer =1./(1+exp(-net));
        layers_val{i+1}  = layer;
     end
     net = transpose(W{1,layers+1})*layer; %get output layer: 10 nodes
     output = 1./(1+exp(-net));    
end

%training network
function W_out = train_network(input_val,digit,output,W,layers)
    output_num = size(W{1,layers+1},2);
    target = zeros(1,output_num);
    target(digit+1) = 1;
    delta = zeros(1,output_num);
    W_out = W;
    for i = 1: output_num %adjusting weights for output layer
        error = abs(target(i)-output(i));
        delta(i) = output(i)*(1-output(i))*error;
        for j = 1:size(W{1,layers},2)
            change = 0.1*delta(i)*input_val{layers+1}(j);
            W_out{1,layers+1}(j,i)  = W{1,layers+1}(j,i) + change;
        end
    end
    for i = layers:-1:1
        deltas = (delta*transpose(W_out{i+1})).*transpose(input_val{i+1}.*(ones(size(input_val{i+1},1),1) - input_val{i+1}));
        delta = deltas;
        for j = 1:size(W{1,i},1)
            change = 0.1*delta*input_val{i}(j);
            for k = 1:size(W{1,i},2)
                W_out{1,i}(j,k) = W{1,i}(j,k) + change(k);
            end
        end
    end
end

function [rate1,rate2] = whole(W,layers,n,train_image,test_image,train_label,test_label)
     W_out = W;
     for i = 1:n
         number = randi([1,n],1);
         input = double(train_image(1:784,number));
         [layers_val,output] = build_network(input,W_out,layers);
         digit = double(train_label(1,number));
         W_out = train_network(layers_val,digit,output,W_out,layers);
     end
     result1 = zeros(60000,1);
     rate1 = 0;
     for i = 1:60000
         digit = double(train_label(1,i));
         input = double(train_image(1:784,i));
         [layers_val,output] = build_network(input,W_out,layers);
         result1(i) = find(max(output));
         if result1(i) ~= digit
             rate1 = rate1+1;
         end
     end
     result2 = zeros(10000,1);
     rate2 = 0;
     for i = 1:10000
         digit = double(test_label(1,i));
         input = double(test_image(1:784,i));
         [layers_val,output] = build_network(input,W_out,layers);
         result2(i) = find(max(output));
         if result2(i) ~= digit
             rate2 = rate2+1;
         end
     end
     
end

