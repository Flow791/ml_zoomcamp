import numpy as np
import tensorflow.lite as tflite

def predict(input_data, model_path='model_2024_hairstyle.tflite'):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    input_data = input_data.astype(np.float32)
    interpreter.set_tensor(input_index, input_data)

    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    return preds
