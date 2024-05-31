import random
import numpy as np
import pyautogui
import cv2
from PIL import Image
from mss import mss
import time
from os import listdir, path, makedirs
from concurrent.futures import ThreadPoolExecutor, as_completed

class Agent:
    def __init__(self):
        self.width, self.height = pyautogui.size()
        print(f"Screen size: {self.width}x{self.height}")

        self.weights = np.random.uniform(-1, 1, (self.height, self.width)).astype(np.float32)
        print("Weights generated\n")
        self.biases = np.random.uniform(-1, 1, (self.height, self.width)).astype(np.float32)
        print("Biases generated\n")
        
        self.activations = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": lambda x: np.tanh(x),
            "relu": lambda x: np.maximum(0, x),
            "argmax": lambda x: np.argmax(x)
        }

        self.flatten = lambda x: [item for sublist in x for item in sublist]

        self.model_loc = "Models/1.20.1/"
        self.images_loc = "Images/Minecraft/1.20.1/"
        self.recorded_images_loc = "Images/Recorded/1.20.1/"
        self.training_data_loc = "Videos/1.20.1/training/"
        self.testing_data_loc = "Videos/1.20.1/testing/"

        self.task_master = self.TaskMaster(self)
        print("Task Master Created\n")
        self.vision_analyzer = self.VisionAnalyzer(self)
        print("Vision Analyzer Created\n")
        self.active_task = None
        self.output = None
    
    def save_model(self, model, model_name):
        makedirs(self.model_loc, exist_ok=True)
        
        weights_path = path.join(self.model_loc, model_name + "_weights.txt")
        biases_path = path.join(self.model_loc, model_name + "_biases.txt")

        if not path.exists(weights_path): 
            with open(weights_path, 'w+') as wf:
                for layer in model:
                    for perceptron in model[layer]:
                        wf.write(" ".join(map(str, perceptron["weights"])) + "\n")
        
        if not path.exists(biases_path):
            with open(biases_path, 'w+') as bf:
                for layer in model:
                    for perceptron in model[layer]:
                        bf.write(" ".join(map(str, perceptron["biases"])) + "\n")

            print(f"Model {model_name} saved")
        
        else:
            print(f"Aborting... Model {model_name} already exists")

    def load_model(self, model_name):
        model = self.create_model(None, ["Input", "hiddenInput", "hiddenOutput", "Output"])

        weights_path = path.join(self.model_loc, model_name + "_weights.txt")
        biases_path = path.join(self.model_loc, model_name + "_biases.txt")

        with open(weights_path, 'r') as wf:
            weight_lines = wf.readlines()

        with open(biases_path, 'r') as bf:
            bias_lines = bf.readlines()

        weight_idx = 0
        bias_idx = 0

        for layer in model:
            for perceptron in model[layer]:
                perceptron["weights"] = list(map(float, weight_lines[weight_idx].strip().split()))
                perceptron["biases"] = list(map(float, bias_lines[bias_idx].strip().split()))
                weight_idx += 1
                bias_idx += 1

        print(f"Model loaded from {model_name}")
        return model
     
    def activated_matmul(self, inputs, weights, biases, activation):
        res = np.dot(inputs, weights) + biases
        return self.activations[activation](res)

    def create_fully_connected_mesh_perceptron(self, inputs, weights, biases, activation, forward_connections, backward_connections):
        perceptron = {
            "inputs": inputs,
            "weights": weights.tolist() if not isinstance(weights, list) else weights,
            "biases": biases.tolist() if not isinstance(biases, list) else biases,
            "activation": activation,
            "forward_connections": forward_connections,
            "backward_connections": backward_connections,
            "loss": [],
        }
        return perceptron
       
    def create_model(self, inputs, layers):
        print("Creating Model...")
        model = {layer: [] for layer in layers}
        
        for idx, layer in enumerate(layers):
            print(f"Creating {layer} Layer with {self.width * self.height} Perceptrons")
            for i in range(self.height):
                for w in range(self.width):
                    perceptron = self.create_fully_connected_mesh_perceptron(inputs, self.weights[i][w], self.biases[i][w], 
                        activation="relu" if idx % 1 and layer != "Output" \
                            else "sigmoid" if idx % 2 and layer != "Output" \
                            else "argmax" if layer == "Output" \
                            else None, 
                        forward_connections=[],
                        backward_connections=[],
                    )

                    model[layer].append(perceptron)

            print(f"{layer} Layer Created with {len(model[layer])} Perceptrons\n")
        
        print("Establishing Fully Connected Layers...")
        connect = lambda layer, connection_type: [layer[p].get(connection_type).append(layer[p + 1]) for p in range(len(layer) - 1)] 
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _, slayys in model.items():
                futures.append(executor.submit(connect, slayys, "forward_connections"))
                futures.append(executor.submit(connect, slayys[::-1], "backward_connections"))

            for future in as_completed(futures):
                future.result()

        print("Model Created and Connections Established")
         
        return model

    def forward(self, model):
        pass

    def backprop(self, model):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def calc_fitness(self):
        pass

    def run_action(self, action):
        if action:
            if action[0] == "move":
                pyautogui.keyDown(action[1])
                time.sleep(action[2])
                pyautogui.keyUp(action[1])
            elif action[0] == "mine":
                pyautogui.mouseDown(button='left')
                time.sleep(action[1])
                pyautogui.mouseUp(button='left')
            elif action[0] == "attack":
                pyautogui.mouseDown(button='left')
                time.sleep(action[1])
                pyautogui.mouseUp(button='left')
            elif action[0] == "build":
                pyautogui.mouseDown(button='right')
                time.sleep(action[1])
                pyautogui.mouseUp(button='right')
            elif action[0] == "camera":
                pyautogui.moveTo(action[1], action[2], duration=0.2)
            print(f"Performed Action: {action}")

    def compare_frames(self, frame1, frame2):
        difference = cv2.absdiff(frame1, frame2)
        result = not np.any(difference)
        print(f"Frames Match: {result}")
        return result

    def train_phase(self, video):
        cap = cv2.VideoCapture(video)
        prev_frame = None
        actions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            if prev_frame is not None:
                action = self.vision_analyzer.identify_action(prev_frame, frame)
                actions.append((action, frame))

            prev_frame = frame

        cap.release()
        print(f"Identified Actions: {actions}")
        self.actions = actions

    def test_phase(self):
        for action, expected_frame in self.actions:
            self.run_action(action)
            time.sleep(1) 
            screenshot = self.vision_analyzer.screenshot()
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            screenshot = cv2.resize(screenshot, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if not self.compare_frames(screenshot, expected_frame):
                print("Action did not match expected result, restarting sequence.")
                break

    class TaskMaster:
        def __init__(self, agent):
            self.active_task = None
            self.completed_tasks = []
            self.routine = []
            self.generated_tasks = []
            self.sub_tasks = {"move": 0, "mine": 0, "attack": 0, "build": 0, "camera": 0, "survive": 0}

            self.agent = agent

            self.inputs = []

            self.training_data = None 

        def calc_immediate_fitness(self):
            pass

        def calc_task_fitness(self):
            pass

        def calc_total_fitness(self, f1, f2):
            return sum((f1 + f2) ** 2 / f2)

        def create_task(self):
            if not self.completed_tasks:
                self.active_task = "survive"
            else:
                self.active_task = random.choice(list(self.sub_tasks.keys()))
                self.routine.append(self.active_task)
                self.generated_tasks.append(self.active_task)
            print(f"Created Task: {self.active_task}")
            return self.active_task
    
    class VisionAnalyzer:
        def __init__(self, agent):
            self.agent = agent
            
            self.block_images = self.load_block_images()
            print("Block Images Established")

            self.inputs = self.preprocess(self.screenshot())
            print("Inputs Established")

            self.model = self.agent.create_model(self.inputs, ["Input", "hiddenInput", "hiddenOutput", "Output"])
            print("Established Model")
            
        def calc_fitness(self, y_pred, y_true):
            pass

        def backprop(self):
            pass

        def preprocess(self, shot):
            shot = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
            shot = cv2.resize(shot, (self.agent.width, self.agent.height), interpolation=cv2.INTER_AREA)
            shot = shot / 255.0

            return shot

        def record(self):
            area = {"top": 0, "left": 0, "width": self.agent.width, "height": self.agent.height}

            with mss() as sct:
                while True:
                    img = Image.frombytes(
                        "RGB",
                        (self.agent.width, self.agent.height),
                        sct.grab(area).rgb
                    )

                    cv2.imshow("window", np.array(img))
                    if cv2.waitKey(60) == ord("q"):
                        cv2.destroyAllWindows()
                        break

            return img

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
                    print(f"Loaded Block Image: {block_name}")

            return block_images

        def identify_blocks(self, screenshot):
            results = []
            processed_screenshot = self.preprocess(screenshot)
            # run model

            # Placeholder for block identification results
            for pixel_data in processed_screenshot:
                block_name = self.predict(pixel_data)
                confidence = random.uniform(0.5, 1.0)
                location = (random.randint(0, self.agent.width), random.randint(0, self.agent.height))
                results.append((block_name, confidence, location))
                print(f"Identified Block: {block_name}, Confidence: {confidence:.2f}, Location: {location}")
            return results

        def predict(self, pixel_data):
            return random.choice(list(self.block_images.keys()))

        def display_results(self, results):
            for block_name, confidence, location in results:
                print(f"Block: {block_name}, Confidence: {confidence:.2f}, Location: {location}")

        def process_frame(self):
            print("Starting Frame Processing...")
            screenshot = self.screenshot()
            results = self.identify_blocks(screenshot)
            self.display_results(results)
            print("Finished Frame Processing.")

        def identify_action(self, prev_frame, curr_frame):
            diff = cv2.absdiff(prev_frame, curr_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            action = None

            if len(contours) > 0:
                for contour in contours:
                    if cv2.contourArea(contour) > 1000: 
                        x, y, w, h = cv2.boundingRect(contour)
                        if x < self.agent.width // 2:
                            if y < self.agent.height // 2: 
                                action = ("move", "w", 0.5) 
                            else:  
                                action = ("move", "s", 0.5)  
                        else:  
                            if y < self.agent.height // 2: 
                                action = ("move", "a", 0.5) 
                            else:  
                                action = ("move", "d", 0.5)  

            if action is None:
                action = ("survive",)

            print(f"Identified Actions: {action}")
            return action

if __name__ == "__main__":
    agent = Agent()
    print("Agent Created\n")
   
    #agent.save_model(agent.vision_analyzer.model, "VisionAnalyzer1")
    #print("vision_analyzer Model Saved\n")
    
    #vision_analyzer = agent.load_model("VisionAnalyzer1")
    #print("vision_analyzer Model Loaded\n")
    
    agent.task_master.create_task()
    agent.train_phase("Videos/1.20.1/training/Vid1.mp4")
    
    #agent.test_phase()

