googlenetLayers = layerGraph(googlenet);
resnetLayers = layerGraph(resnet18);

googlenetLayers = removeLayers(googlenetLayers, {'output'});
resnetLayers = removeLayers(resnetLayers, {'ClassificationLayer_predictions'}); 

for i = 1:numel(resnetLayers.Layers)
    resnetLayers = renameLayer(resnetLayers, resnetLayers.Layers(i).Name, ['resnet_' resnetLayers.Layers(i).Name]);
end

lgraph = addLayers(googlenetLayers, resnetLayers.Layers);

customFC = fullyConnectedLayer(1000, 'Name', 'custom_fc');
softmaxLayer = softmaxLayer('Name', 'custom_softmax');
classificationLayer = classificationLayer('Name', 'custom_classoutput');

lgraph = addLayers(lgraph, [customFC softmaxLayer classificationLayer]);

lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'custom_fc');
lgraph = connectLayers(lgraph, 'custom_fc', 'custom_softmax');
lgraph = connectLayers(lgraph, 'custom_softmax', 'custom_classoutput');

analyzeNetwork(lgraph);
