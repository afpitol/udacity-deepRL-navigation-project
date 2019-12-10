# Learning algorithm  
The solution implements a Deep Q-Networks algorithm that was first demonstrated ins this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).
The Deep Q-Network implements the Q-Learning replacing the Q table (action-values) with a deep network wich allows us to apply the algorithm to very large state environments as well as continuous states environments.
  
  To prevent the network from becoming instable and, therefor, diverging or not learning at all, two aproaches were applied:
  - *Experience Replay*
  - *Fixed Q-targets*
  
  The algorithm is described as follows:
  ![Algorithm](/images/algorithm.PNG)
  
  
 # The architecture
  The architecture used here is based on the one used on the [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf):
  ![Architecture](/images/architecturePNG.PNG)
  
 ### Deep Q-Network  
 For our algorithm we had a Deep Q-Network (defined in model.py), consisted of:
    - Local Q-network and target Q-network (for fixed target implementation): the Q-networks consisted of 3-layered fully connected neural network. The first 2 layers have a RELU activation function.   
    The first layer has an input of _state_size_ and the third layer has an output _action_size_. The hidden layers has default size of 64 units 
  
"_The outputs correspond to the predicted Q-values of the individual actions for the input state. The main advantage of this type of architecture is the ability to compute Q-valuesfor all possible actions in a given state with only a single forward pass through the network_". [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), pag 6, Model Architeture

  ### The agent   
  The agent (defined at dqn_agent.py) was built with two Q-Networks, local and target (for the fixed target strategy), an ADAM optimizer and a ReplayBuffer (for the experience replay strategy).
     
  Both AGENT and ReplayBuffer classes were defined at the dqn_agent.py file as follows:   
  ##### *Hyperparameters*
  BUFFER_SIZE = int(1e5)  # replay buffer size
  BATCH_SIZE = 64         # minibatch size
  GAMMA = 0.99            # discount factor
  TAU = 1e-3              # for soft update of target parameters
  LR = 5e-4               # learning rate 
  UPDATE_EVERY = 4        # how often to update the network
    
  ##### class AGENT   
  ###### *Agent(state_size, action_size, seed)*  
  Initialize an Agent object.  

  ###### *step*(state, action, reward, next_state, done):
  Saves experience on the replay buffer and updates the target networks at every UPDATE_EVERY.
      
  ###### *act*(state, eps=0.):
  Returns actions for given state as per current policy.
  
  ###### *learn*(self, experiences, gamma):
  Updates value parameters using given batch of experience tuples.
  
  ###### *soft_update*(local_model, target_model, tau):
  Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target  
        
          
  ##### *class ReplayBuffer*(action_size, buffer_size, batch_size, seed)  
  Fixed-size buffer to store experience tuples.  
      
  ###### *add*(self, state, action, reward, next_state, done):  
  Add a new experience to memory.  
      
  ###### *sample*(self):  
  Randomly sample a batch of experiences from memory.  
    
  ###### *len*():  
  Return the current size of internal memory.
    
      
   ### Training
   Finally, the training algorithm is defined at the *navigation.ipynb*, and called with the following hyperparameters values:  
   ###### *dqn*(n_episodes=400, eps_end=0.04, eps_decay=0.95)
