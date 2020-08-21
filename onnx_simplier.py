import onnx
from onnxsim import simplify
path="D:\\doc\\github_program_month_8\\Pytorch_Retinaface\\"
model_name="FaceDetector"
# load your predefined ONNX model
model = onnx.load(path + model_name + '.onnx')

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"