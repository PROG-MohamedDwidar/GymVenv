# GymVenv
Simulation environmet for reinforcement learning  
  
  
  
# to avoid the __version__ error remeber:  
-In Python310\lib\site-packages\rl\callbacks.py change from tensorflow.keras to from keras import __version__. that should do the trick.  
-In Python310\lib\site-packages\rl\util.py change optimizer._name to optimizer.name    
  
## don't use keras-rl2 it is discontinued and it is the cause of the problems
