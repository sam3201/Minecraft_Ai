import numpy as np
import pyautogui
import cv2
from mss import mss
import time
from os import listdir, path, makedirs 
from concurrent.futures import ThreadPoolExecutor, as_completed

class Agent:
    def __init__(self, vision_model_name=None, action_model_name=None, duration_model_name=None):
        self.width, self.height = pyautogui.size()
        print(f"Screen size: {self.width}x{self.height}")

        self.weights = np.random.uniform(-1, 1, (self.height, self.width)).astype(np.float16)
        print(f"weights_shape: {len(self.weights), len(self.weights[0])}\n")
        self.biases = np.random.uniform(-1, 1, (self.height, self.width)).astype(np.float16)
        print(f"biases_shape: {len(self.biases), len(self.biases[0])}\n")

        self.activations = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": lambda x: np.tanh(x),
            "relu": lambda x: np.maximum(0, x),
            "argmax": lambda x: np.argmax(x),
            "beta-max": lambda x, lst: np.argmax(lst) if np.argmax(x) == 1 else 0
        }
       
        self.activations_derivatives = {
                "sigmoid": lambda x: x * (1 - x),
                "tanh": lambda x: 1 - x ** 2,
                "relu": lambda x: np.where(x > 0, 1, 0),
                "argmax": lambda x: np.where(x > 0, 1, 0)
        }

        self.losses = {
            "mse": lambda y_true, y_pred: np.mean(np.square(y_true - y_pred)),
            "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
            "cross_entropy": lambda y_true, y_pred: np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)),
            "binary_cross_entropy": lambda y_true, y_pred: np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)),
            "poisson": lambda y_true, y_pred: np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)),
            "hinge": lambda y_true, y_pred: np.mean(np.maximum(0, 1 - y_true * y_pred)),
            "squared_hinge": lambda y_true, y_pred: np.mean(np.maximum(0, 1 - y_true * y_pred) ** 2),
            "logcosh": lambda y_true, y_pred: np.mean(np.log(np.cosh(y_true - y_pred))),
            "squared_logcosh": lambda y_true, y_pred: np.mean(np.square(np.log(np.cosh(y_true - y_pred)))),
            "categorical_hinge": lambda y_true, y_pred: np.mean(np.maximum(0, 1 - y_true * y_pred)),
            "categorical_crossentropy": lambda y_true, y_pred: np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)),
            "sparse_categorical_crossentropy": lambda y_true, y_pred: np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)),
            "kullback_leibler_divergence": lambda y_true, y_pred: np.mean(y_true * np.log(y_true / y_pred) + (1 - y_true) * np.log((1 - y_true) / (1 - y_pred))),
            "cosine_proximity": lambda y_true, y_pred: np.mean(y_true - y_pred),

        }

        self.losses_derivatives = {
            "mse": lambda y_true, y_pred: 2 * (y_true - y_pred),
            "mae": lambda y_true, y_pred: np.where(y_true > y_pred, 1, -1),
            "cross_entropy": lambda y_true, y_pred: y_pred - y_true,
            "binary_cross_entropy": lambda y_true, y_pred: y_pred - y_true,
            "poisson": lambda y_true, y_pred: y_pred - y_true,
            "hinge": lambda y_true, y_pred: np.where(y_true * y_pred < 1, -y_true, 0),
                "squared_hinge": lambda y_true, y_pred: np.where(y_true * y_pred < 1, -2 * y_true * y_pred, 0),
                "logcosh": lambda y_true, y_pred: 1 / np.cosh(y_true - y_pred),
                "squared_logcosh": lambda y_true, y_pred: 1 / np.cosh(y_true - y_pred) ** 2,
                "categorical_hinge": lambda y_true, y_pred: np.where(y_true * y_pred < 1, -y_true, 0),
                "categorical_crossentropy": lambda y_true, y_pred: y_pred - y_true,
                "sparse_categorical_crossentropy": lambda y_true, y_pred: y_pred - y_true,
                "kullback_leibler_divergence": lambda y_true, y_pred: y_pred - y_true,
                "cosine_proximity": lambda y_true, y_pred: y_pred - y_true,
        }

        self.flatten = lambda x: [item for sublist in x for item in sublist]

        self.model_loc = "Models/1.20.1/"
        self.images_loc = "Images/Minecraft/1.20.1/"
        self.recorded_images_loc = "Images/Recorded/1.20.1/"
        self.training_data_loc = "Videos/1.20.1/training/"
        self.testing_data_loc = "Videos/1.20.1/testing/"
       
        self.vision_analyzer = self.VisionAnalyzer(self, vision_model_name)
        print("Vision Analyzer Created\n")
       
        self.action_analyzer = self.ActionAnalyzer(self, action_model_name, duration_model_name)
        print("Action Analyzer Created\n")
         
    def save_model(self, model, model_name):
        model_dir = path.join(self.model_loc, model_name)
        try:
            makedirs(model_dir, exist_ok=True)

            weights_path = path.join(model_dir, model_name + "_weights.txt")
            biases_path = path.join(model_dir, model_name + "_biases.txt")

            with open(weights_path, 'w+') as wf, open(biases_path, 'w+') as bf:
                for layer in model:
                    for perceptron in model[layer]:
                        perceptron["weights"] = float(perceptron["weights"]) 
                        perceptron["biases"] = float(perceptron["biases"]) 
                        wf.write(str(perceptron["weights"]) + "\n")
                        bf.write(str(perceptron["biases"]) + "\n")

            print(f"Model {model_name} saved")
            return model
        except Exception as e:
            print(f"Error saving model {model_name}: {e}")
            return None


    def load_model(self, model_name):
        try:
            model_dir = path.join(self.model_loc, model_name)
        except Exception as e:
            print(f"Model {model_name} does not exist.")
            return None

        model = self.create_model(None, ["Input", "hiddenInput", "hiddenOutput", "Output"])

        weights_path = path.join(model_dir, model_name + "_weights.txt")
        if not path.exists(weights_path):
            print(f"Model {model_name} does not have weights.")
            return None

        biases_path = path.join(model_dir, model_name + "_biases.txt")
        if not path.exists(biases_path):
            print(f"Model {model_name} does not have biases.")
            return None

        with open(weights_path, 'r') as wf, open(biases_path, 'r') as bf:
            weight_lines = wf.readlines()
            bias_lines = bf.readlines()

        for idx, layer in enumerate(model):
            for perceptron in model[layer]:
                perceptron["weights"] = float(weight_lines[idx].strip())
                perceptron["biases"] = float(bias_lines[idx].strip())
        
        print(f"Model loaded from {model_name}")
        return model

    def create_fully_connected_mesh_perceptron(self, inputs, weights, biases, activation, forward_connections, backward_connections):
        perceptron = {
            "inputs": inputs,
            "weights": weights,
            "biases": biases, 
            "activation": activation,
            "forward_connections": forward_connections,
            "backward_connections": backward_connections,
        }
        return perceptron
       
    def create_model(self, inputs, layers):
        print("Creating Model...")
        model = {layer: [] for layer in layers}
        
        for idx, layer in enumerate(layers):
            print(f"Creating {layer} Layer with {self.width * self.height} Perceptrons")
            for i in range(self.height):
                for w in range(self.width):
                    if layer == "Output":
                        activation = self.activations["argmax"]
                    elif idx % 2 == 0:
                        activation = self.activations["sigmoid"]
                    else:
                        activation = self.activations["relu"]
                    
                    perceptron = self.create_fully_connected_mesh_perceptron(
                        inputs, 
                        self.weights[i][w], 
                        self.biases[i][w], 
                        activation=activation,
                        forward_connections=[],
                        backward_connections=[],
                    )

                    model[layer].append(perceptron)

            print(f"{layer} Layer Created with {len(model[layer])} Perceptrons\n")
        
        print("Establishing Fully Connected Layers...")
        connect = lambda layer, connection_type: [layer[p].get(connection_type).append(layer[p + 1]) for p in range(len(layer) - 1)] 
        with ThreadPoolExecutor() as executor:
            futures = []
            for _, layer in model.items():
                futures.append(executor.submit(connect, layer, "forward_connections"))
                futures.append(executor.submit(connect, layer[::-1], "backward_connections"))

            for future in as_completed(futures):
                future.result()

        print("Model Created and Connections Established")
         
        return model
     
    def activated_matmul(self, inputs, weights, biases, activation):
        res = np.dot(inputs, weights) + biases 
        return activation(res)

    def forward(self, inputs, neural_entity):
        neural_entity = list(neural_entity.values())
        
        res = []
        layer = 0

        while layer < len(neural_entity):
            perceptron = 0
            while perceptron < len(neural_entity[layer]):
                for i in range(len(inputs)):
                    res.append(self.activated_matmul(inputs[i], neural_entity[layer][perceptron]["weights"], neural_entity[layer][perceptron]["biases"], neural_entity[layer][perceptron]["activation"]))  
                perceptron += 1
            layer += 1
        
        return res

    def backprop(self, neural_entity, inputs, y_true, y_pred, loss): 
        loss_deriv = self.losses_derivatives.get(loss.__name__)
        if loss_deriv:
            error_deriv = loss_deriv(y_true, y_pred)

            for layer in reversed(neural_entity):
                new_error_derivs = []
                for perceptron in neural_entity[layer]:
                    derivative_activation = self.activations_derivatives.get(perceptron["activation"].__name__)
                    if derivative_activation: 
                        delta = error_deriv * derivative_activation(y_pred)
                        weights_deriv = delta * derivative_activation(inputs)
                        
                        perceptron["weights"] -= weights_deriv 
                        perceptron["biases"] -= delta
                        new_error_derivs.append(delta)
                    else:
                        raise Exception("Invalid Activation or Loss Function")
                error_deriv = np.sum(new_error_derivs)
        else:
            raise Exception("Invalid Loss Function")

        return error_deriv 

    def run_action(self, task):
        action = None 
        match task["name"]:
            case "left_click":
                action = pyautogui.click(button="left", duration=task["Duration"])
        
            case "right_click":
                action = pyautogui.click(button="right", duration=task["Duration"])

            case "mouse_move_x":
                action = pyautogui.moveTo(task["Name"], None) 

            case "mouse_move_y":
                action = pyautogui.moveTo(task["Name"], None)

            case _:
                action = lambda: pyautogui.press(task["Name"])
        
        if callable(action):
            start_time = time.time()
            while time.time() - start_time < task["Duration"]:
                action()
        
        print(f"Performed Action: {action}")
        
        return action

    def compare_frames(self, frame1, frame2):
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        diff = cv2.absdiff(frame1, frame2)
        thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) 
        non_zero = cv2.countNonZero(thresh)
        total = thresh.size  
        similarity = non_zero / total 
        
        print(f"Frames Match: {similarity}")
        return similarity 


    def train_phase(self, video, output_file, frames_dir):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Error opening video file: {video}")
            return []

        prev_frame = None
        actions = []

        frames_dir = path.join(self.training_data_loc, frames_dir)
        output_file = path.join(self.training_data_loc, output_file)

        if not path.exists(frames_dir):
            makedirs(frames_dir)

        processed_frames = 0
        if path.exists(output_file):
            with open(output_file, 'r') as f:
                processed_frames = len(f.readlines())
            f.close()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")
        print(f"Starting from frame: {processed_frames + 1}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, processed_frames)

        try:
            with open(output_file, 'a') as f:
                frame_count = processed_frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Failed to read frame at position {frame_count}")
                        break

                    frame_count += 1
                    print(f"Processing frame {frame_count} of {total_frames}")

                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                        if prev_frame is not None:
                            action = self.vision_analyzer.identify_action(prev_frame, frame)
                            f.write(f"{action}\n")
                            actions.append(action)

                            frame_filename = path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                        prev_frame = frame
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                        cap.release()
                        f.close()
                        return
        except Exception as e:
            print(f"Error during training phase: {e}")
            return
        finally:
            cap.release()
            print("Action Identification Complete")

        return actions

    def train_predictions(self, output_dir, testing_data, testing_frames, action_model, duration_model):
        output_dir = path.join(self.training_data_loc, output_dir)
        if not path.exists(output_dir):
            makedirs(output_dir)
        else:
            print("Output Dir Should Not Exist. Remove and Try Again") 
            return
        
        testing_data = path.join(self.training_data_loc, testing_data)
        if not path.exists(testing_data):
            makedirs(testing_data)
        else:
            print("Testing Data Should Not Exist. Remove and Try Again")
            return

        testing_frames = path.join(self.training_data_loc, testing_frames)
        if not path.exists(testing_frames):
            print(f"{testing_frames} does not exist")
            return

        action_model = path.join(self.training_data_loc, action_model)
        if not path.exists(action_model):
            print(f"{action_model} does not exist")
            return

        duration_model = path.join(self.training_data_loc, duration_model)
        if not path.exists(duration_model):
            print(f"{duration_model} does not exist")
            return
        
        processed_actions = 0
        total_actions = len(open(action_model).readlines())
        print(f"Total Action in Predictions: {total_actions}")
        print(f"Starting from action: {processed_actions + 1}")

        with open(testing_data, 'r') as identified_actions, open(testing_frames, 'r') as identified_frames:
            for action, frme in zip(identified_actions, identified_frames): 
                action = action.strip()
                frme = frme.strip()

                cap = cv2.VideoCapture(0)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_actions += 1
                    print(f"Processing frame {processed_actions} of {total_actions}")

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                    if frame is not None:
                        while True:
                            task, action_model_inputs, duration_model_inputs = self.action_analyzer.create_task(action, frame)
                            self.run_action(task)
                           
                            y_true = 1
                            y_pred = self.compare_frames(frme, frame) 
                            loss = self.losses["mse"](y_true, y_pred)

                            if loss >= 0.1:
                                self.backprop(self.action_analyzer.action_model, action_model_inputs, y_true, y_pred, loss) 
                                self.backprop(self.action_analyzer.duration_model, duration_model_inputs, y_true, y_pred, loss) 
                            else:
                                cap.release()
                                with open(path.join(output_dir, "generalized_actions/"), 'a') as f:
                                    action_filename = path.join(output_dir, f"action_{processed_actions:04d}.jpg")
                                    cv2.imwrite(action_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                                    f.write(f"{action}\n")
                                break

    def test_phase(self):
        while True:
            screenshot = self.vision_analyzer.screenshot()
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            screenshot = cv2.resize(screenshot, (self.width, self.height), interpolation=cv2.INTER_AREA)

    class ActionAnalyzer:
        def __init__(self, agent, action_model_name, duration_model_name):
            self.current_action = None
            self.complete_actions_frames = {}
            self.iterations = 0

            self.key_presses = {
                "w": 0, "a": 0, "s": 0, "d": 0, "e": 0, "f": 0, "g": 0, "h": 0, "j": 0, "k": 0, "l": 0,
                "i": 0, "u": 0, "o": 0, "p": 0, "q": 0, "r": 0, "t": 0, "y": 0, "u": 0, "z": 0, "x": 0,
                "c": 0, "v": 0, "b": 0, "n": 0, "m": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0,
                "7": 0, "8": 0, "9": 0, "0": 0, "shiftleft": 0, "ctrlleft": 0, "space": 0, "escape": 0,
                "left_click": 0, "right_click": 0, "mouse_move_x": 0, "mouse_move_y": 0
            }

            self.agent = agent
            self.training_data = None 

            if action_model_name:
                self.action_model = self.agent.load_model(action_model_name) 
                if self.action_model is None:
                    self.action_model = self.agent.create_model(None, ["Input", "hiddenInput", "hiddenOutput", "Output"]) 
                    self.agent.save_model(self.action_model, action_model_name)
                    print(f"Model {action_model_name} created\n")
                else:
                    print(f"Model {action_model_name} loaded\n")
            else:
                print("Name Not Provided. Provide name for action model version\n")
                return 

            if duration_model_name:
                self.duration_model = self.agent.load_model(duration_model_name) 
                if self.duration_model is None:
                    self.duration_model = self.agent.create_model(None, ["Input", "hiddenInput", "hiddenOutput", "Output"]) 
                    self.agent.save_model(self.duration_model, duration_model_name)
                    print(f"Model {duration_model_name} created\n")
                else:
                    print(f"Model {duration_model_name} loaded\n")
            else:
                print("Name Not Provided. Provide name for action model version\n")
                return  

        def create_task(self, identified_action, frame):
            task = {"Name": None, "Duration": 0}

            action_idx = np.sum(self.agent.forward([identified_action, frame], self.action_model))
            task["Name"] = list(self.key_presses.keys())[action_idx]

            task["Duration"] = self.agent.forward([identified_action, frame, action_idx], self.duration_model)
            
            return task, [identified_action, frame], [identified_action, frame, action_idx]

    class VisionAnalyzer:
        def __init__(self, agent, model_name):
            self.agent = agent
            self.block_images = self.load_block_images()
            print("Block Images Established")
            
            if model_name:
                self.model = self.agent.load_model(model_name)
                if self.model is None:
                    self.model = self.agent.create_model(None, ["Input", "hiddenInput", "hiddenOutput", "Output"])
                    self.agent.save_model(self.model, model_name)
                    print(f"Model {model_name} created\n")
                else:
                    print(f"Model {model_name} loaded\n")
            else:
                print("Name Not Provided. Provide name for vision model version\n")
                return
    
            self.actions = ["survive", "attack", "heal", "build", "craft", "mine"]

        def preprocess(self, shot):
            shot = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
            shot = cv2.resize(shot, (self.agent.width, self.agent.height), interpolation=cv2.INTER_AREA)
            shot = shot / 255.0
            return shot

        def screenshot(self):
            area = {"top": 0, "left": 0, "width": self.agent.width, "height": self.agent.height}
            with mss() as sct:
                screenshot = np.array(sct.grab(area))
            return screenshot

        def load_block_images(self):
            block_images = {}
            for filename in listdir(self.agent.images_loc):
                if filename.endswith(".png"):
                    block_name = filename[:-4]
                    block_images[block_name] = cv2.imread(path.join(self.agent.images_loc, filename), cv2.IMREAD_UNCHANGED)
            return block_images

        def identify_action(self, prev_frame, curr_frame):
            try:
                diff = cv2.absdiff(prev_frame, curr_frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                
                action = None
               
                processed_shot = self.preprocess(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)) 

                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.agent.forward, processed_shot, self.model)]

                for future in as_completed(futures):
                    action_partials = future.result()

                action = self.agent.activations["beta-max"](np.sum(action_partials), self.actions)
                
                if action is None:
                    action = ("survive",)

                print(f"Identified Actions: {action}")
                return action
            except Exception as e:
                print(f"Error identifying action: {e}")
                import traceback
                traceback.print_exc()
                return None

def run(agent=None, model_names=None, video_source=None, phase="Train", output_dir=None):
    if agent is None:
        print("Agent not provided")
        return None
    
    if model_names is None:
        print("Model names not provided")
        return None
   
    if phase == "Train":
        if video_source is None:
            print("Video source not provided for training phase")
            return None
        return agent.train_phase(video_source, "output.txt", "frames")
    
    elif phase == "Train_Predictions":
        if output_dir is None:
            print("Output directory not provided for training predictions phase")
            return None
        if not all([agent.action_analyzer.action_model, agent.action_analyzer.duration_model]):
            print("Action model or duration model not loaded")
            return None
        return agent.train_predictions(output_dir, "output.txt", "frames", agent.action_analyzer.action_model, agent.action_analyzer.duration_model)
    
    elif phase == "Test":
        return agent.test_phase()
    
    else:
        print(f"Unknown phase: {phase}")
        return None

if __name__ == "__main__": 
    vision_analyzer_name = "VisionAnalyzer1"
    action_analyzer_name = "ActionAnalyzer1"
    duration_model_name = "DurationModel1"

    agent = Agent(vision_model_name=vision_analyzer_name, action_model_name=action_analyzer_name, duration_model_name=duration_model_name) 
    time.sleep(1)

    video_source = "Videos/1.20.1/training/Vid1.mp4"
   
    try:
        predicted_actions = run(agent=agent, model_names=[vision_analyzer_name], video_source=video_source, phase="Train")
    except Exception as e:
        print(f"Error during training phase: {e}")
    finally:
        if agent:
            agent.save_model(agent.vision_analyzer.model, vision_analyzer_name)
            print("Vision Model Saved")

    """
    try:
        trained_predictions = run(agent=agent, model_names=[vision_analyzer_name, action_analyzer_name, duration_model_name], video_source=video_source, phase="Train_Predictions", output_dir="output_dir") 
    except Exception as e:
        print(f"Error during training predictions phase: {e}")
    finally:
        if agent:
            agent.save_model(agent.action_analyzer.action_model, action_analyzer_name)
            agent.save_model(agent.action_analyzer.duration_model, duration_model_name) 
            print("Action Model Saved")
    """
