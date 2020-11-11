import tensorflow as tf

class config():
  

    # Uncomment to train on different environments
    #env_name="CartPole-v0"
    #env_name="InvertedPendulum-v1"
    #env_name="HalfCheetah-v1"
    #env_name="Walker2d-v1"
    #env_name="Humanoid-v1"
    #env_name="Hopper-v1"
    #env_name="Ant-v1"
    #env_name="InvertedDoublePendulum-v1"
    #env_name="HumanoidStandup-v1"
    

    record           = True 

    # output config
    output_path  = "results/" + env_name + "/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5 
    summary_freq = 1

    
    # model and training config
    num_batches = 100 #200 # number of batches trained on 
    batch_size =50000 #50000(halfcheetah,humanoid,ant,hopper,walker,HumanoidStandup) #1000(inverted pendulum,inverted double pendulum and cartpole) # number of steps used to compute each policy update
    max_ep_len = 1000 #1000 # maximum episode length
    learning_rate = 3e-2 #3e-2
    gamma = 0.9 #0.9(halfcheetah,humanoid,ant,hopper,walker,HumanoidStandup) #0.99(inverted pendulum,inverted double pendulum and cartpole) # the discount factor 
    use_baseline = True 
    normalize_advantage=True 
    
    # parameters for the policy and baseline models
    n_layers = 3#3(halfcheetah,humanoid,ant,hopper,walker,HumanoidStandup)#1(cartpole,inverted double pendulum and inverted pendulum) 
    layer_size =32 #32(halfcheetah,humanoid,ant,hopper,walker,HumanoidStandup) #16(cartpole,inverted double pendulum and inverted pendulum)  
    activation = tf.nn.relu #tf.nn.relu(halfcheetah,humanoid,ant,inverted double pendulum,hopper,walker,HumanoidStandup) #tf.nn.selu(cartpole and inverted pendulum)  


    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0:
        max_ep_len = batch_size