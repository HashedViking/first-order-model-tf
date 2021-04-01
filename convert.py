import coremltools as ct

### SavedModel/TFLite to CoreML
name = 'generator'
model = ct.convert(
    './saved_models/vox/' + name,
    source="tensorflow"
)
model.save(name + '.mlmodel')
print('Done.')

### ONNX to CoreML
name = 'generator'
model = ct.converters.onnx.convert(model=name+'.onnx', minimum_ios_deployment_target='13')
model.save(name + '.mlmodel')
print('Done.')

### SavedModel/TFLite to ONNX
# python -m tf2onnx.convert --saved-model "./saved_models/vox/generator" --output generator.onnx