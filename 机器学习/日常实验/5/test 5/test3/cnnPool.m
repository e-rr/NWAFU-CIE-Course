function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);
    for imageNum = 1:numImages
        for featureNum = 1:numFilters
            featuremap = squeeze(convolvedFeatures(:,:,featureNum,imageNum));
            pooledFeaturemap = conv2(featuremap,ones(poolDim)/(poolDim^2),'valid');
            pooledFeatures(:,:,featureNum,imageNum) = pooledFeaturemap(1:poolDim:end,1:poolDim:end);
        end
    end
end

