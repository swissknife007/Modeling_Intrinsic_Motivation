from multiprocessing import Process
import numpy as np

class EmulatorRunner(Process):

    def __init__(self, worker_id, emulators, variables, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.worker_id = worker_id
        self.emulators = emulators
        self.variables = variables
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()

    def _run(self):
        count = 0
        while True:
        
            instruction = self.queue.get()
            if instruction is None:
                break
            
            shared_states = self.variables[0]
            shared_rewards = self.variables[1]
            shared_episode_over = self.variables[2]
            shared_actions = self.variables[3]
            
            for i, (emulator, action) in enumerate(zip(self.emulators, self.variables[-1])):            
                emulator = self.emulators[i]
                action = shared_actions[i]
                #print ('action:', action)
                # new_s, reward, episode_over = emulator.next(action)
                new_s, reward, episode_over,_ = emulator.step(np.argmax(action))
                new_s = new_s * 255
                #print ('new_s.shape:', new_s.shape)
                #print ('new_s:', new_s)
                if episode_over:
                    # shared_states[i] = emulator.get_initial_state()
                    shared_states[i] = 255 * emulator.reset()
                else:
                    shared_states[i] = new_s            
                shared_rewards[i] = reward
                shared_episode_over[i] = episode_over
            
            count += 1
            
            # barrier is a queue shared by all workers
            # when a worker is done executing actions for envs it manages
            # it puts True to barrier which later should be 
            self.barrier.put(True)



