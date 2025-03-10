import numpy as np
import torch.nn as nn

NUM_DIALOGUE = 200 #1017 

def restructure_dialogue_padding(dones, states, actions, next_states):
    def restructure_dialogue(t_dones, states, actions, next_states):
        # Restructure a new dialogue
        dialogue = [[] for _ in range(len(t_dones))]
        ii = 0
        start_index = 0
        dialogue[ii].append(states[0])
        for i in range(len(t_dones)):
            start_index += 1
            dialogue[ii].append(actions[i])
            dialogue[ii].append(next_states[i])
            if t_dones[i] != 0 and i + 1 < len(t_dones):
                ii += 1
                dialogue[ii].append(states[i + 1])
        return dialogue
    
    # Padding dialogues
    maxlen = 25
    dialogue = restructure_dialogue(dones, states, actions, next_states)
    dialogue = [dialog for dialog in dialogue if len(dialog)>1] 


    # Convert each dialogue to a numpy array of shape (len(dialogue), 768)
    dialogues = [np.stack(seq) for seq in dialogue[:NUM_DIALOGUE]]
    
    # Pad each dialogue to a length of maxlen (25)
    padded_dialogues = []
    for seq in dialogues:
        if seq.shape[0] < maxlen:
            pad_size = maxlen - seq.shape[0]
            pad = np.zeros((pad_size, seq.shape[1]))
            padded_seq = np.concatenate([seq, pad], axis=0)
        else:
            padded_seq = seq[:maxlen]
        padded_dialogues.append(padded_seq)
    
    pad_dialogue = np.stack(padded_dialogues, axis=0)
    
    return pad_dialogue

