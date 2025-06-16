function testnetwork(net, image)

I = imread(image);
R = imresize(I, [224, 224]);

if size(R, 3) == 1
        R = cat(3, R, R, R);  % Convert grayscale to RGB
    end

[Label, Probability] = classify(net, R);

figure;
imshow(R);
title({char(Label), num2str(max(Probability), 6) })

end