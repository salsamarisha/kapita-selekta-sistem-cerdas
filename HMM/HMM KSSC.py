
# coding: utf-8

# In[2]:


import Bio


# In[3]:


from Bio import Alphabet
from Bio.Seq import MutableSeq
from Bio.Seq import Seq

from Bio.HMM import MarkovModel
from Bio.HMM import Trainer
from Bio.HMM import Utilities


# In[4]:


print(Bio.__version__)


# In[49]:


from Bio.Seq import Seq


# In[6]:


import Bio.HMM.MarkovModel as hmm


# In[7]:


class DiceRollAlphabet(Alphabet.Alphabet):
    letters = ['A','T','C','G']

class DiceTypeAlphabet(Alphabet.Alphabet):
    letters = ['M','I','D']


# In[8]:


mm_builder = hmm.MarkovModelBuilder(DiceTypeAlphabet(),DiceRollAlphabet())


# In[9]:


mm_builder.allow_all_transitions()
mm_builder.set_random_probabilities()


# In[10]:


print(mm_builder)


# In[11]:


DiceTypeAlphabet()


# In[12]:


mm_builder.set_emission_score('I', 'A', .1)
mm_builder.set_emission_score('I', 'T', .1)
mm_builder.set_emission_score('I', 'C', .1)
mm_builder.set_emission_score('I', 'G', .1)
mm_builder.set_emission_score('D', 'A', .1)
mm_builder.set_emission_score('D', 'T', .1)
mm_builder.set_emission_score('D', 'C', .1)
mm_builder.set_emission_score('D', 'G', .1)
mm_builder.set_emission_score('M', 'A', .8)
mm_builder.set_emission_score('M', 'T', .8)
mm_builder.set_emission_score('M', 'C', .8)
mm_builder.set_emission_score('M', 'G', .8)


# In[191]:


standard_mm = mm_builder.get_markov_model()


# In[32]:


def _loaded_dice_roll(chance_num, cur_state):
    """Generate a loaded dice roll based on the state and a random number
    """

    if cur_state == 'M':
        if chance_num <= (float(1) / float(6)):
            return 'A'
        elif chance_num <= (float(2) / float(6)):
            return 'T'
        elif chance_num <= (float(3) / float(6)):
            return 'C'
        else:
            return 'G'
    elif cur_state == 'I':
        if chance_num <= (float(1) / float(6)):
            return 'A'
        elif chance_num <= (float(2) / float(6)):
            return 'T'
        elif chance_num <= (float(3) / float(6)):
            return 'C'
        else:
            return 'G'
    elif cur_state=='D':
        if chance_num <= (float(1) / float(6)):
            return 'A'
        elif chance_num <= (float(2) / float(6)):
            return 'T'
        elif chance_num <= (float(3) / float(6)):
            return 'C'
        else:
            return 'G'    
    else:
        raise ValueError("Unexpected cur_state %s" % cur_state)
    '''        
    if cur_state == 'A':
        if chance_num <= .8:
            return 'M'
        elif chance_num <= .2:
            return 'I'
        else:
            return 'D'
    if cur_state == 'T':
        if chance_num <= .8:
            return 'M'
        elif chance_num <= .2:
            return 'I'
        else:
            return 'D'
    if cur_state == 'C':
        if chance_num <= .8:
            return 'M'
        elif chance_num <= .2:
            return 'I'
        else:
            return 'D'
    if cur_state == 'G':
        if chance_num <= .8:
            return 'M'
        elif chance_num <= .2:
            return 'I'
        else:
            return 'D'
    else:
        raise ValueError("Unexpected cur_state %s" % cur_state)
    '''


# In[33]:


import random
def generate_rolls(num_rolls):
    cur_state = 'M'
    roll_seq = MutableSeq('', DiceRollAlphabet())
    state_seq = MutableSeq('', DiceTypeAlphabet())

    # generate the sequence
    for roll in range(num_rolls):
        state_seq.append(cur_state)
        
        chance_num = random.random()

        new_roll = _loaded_dice_roll(chance_num, cur_state)
        roll_seq.append(new_roll)

        chance_num = random.random()
        if cur_state == 'A':
            if chance_num <= .8:
                cur_state = 'A'
        elif cur_state == 'A':
            if chance_num <= .5:
                cur_state = 'T'
        elif cur_state == 'A':
            if chance_num <= .4:
                cur_state = 'C'
        elif cur_state == 'A':
            if chance_num <= .3:
                cur_state = 'G'
        elif cur_state == 'T':
            if chance_num <= .8:
                cur_state = 'T'
        elif cur_state == 'T':
            if chance_num <= .5:
                cur_state = 'A'
        elif cur_state == 'T':
            if chance_num <= .4:
                cur_state = 'C'
        elif cur_state == 'T':
            if chance_num <= .3:
                cur_state = 'G'
        elif cur_state == 'C':
            if chance_num <= .8:
                cur_state = 'C'
        elif cur_state == 'C':
            if chance_num <= .5:
                cur_state = 'A'
        elif cur_state == 'C':
            if chance_num <= .4:
                cur_state = 'T'
        elif cur_state == 'C':
            if chance_num <= .3:
                cur_state = 'G'
        elif cur_state == 'G':
            if chance_num <= .8:
                cur_state = 'G'
        elif cur_state == 'G':
            if chance_num <= .5:
                cur_state = 'A'
        elif cur_state == 'G':
            if chance_num <= .4:
                cur_state = 'C'
        elif cur_state == 'G':
            if chance_num <= .3:
                cur_state = 'T'

    return roll_seq.toseq(), state_seq.toseq()


# In[39]:


baum_welch_mm = mm_builder.get_markov_model()
standard_mm = mm_builder.get_markov_model()

rolls, states = generate_rolls(10)


# In[37]:


VERBOSE = 0

def stop_training(log_likelihood_change, num_iterations):
    """Tell the training model when to stop.
    """
    if VERBOSE:
        print("ll change: %f" % log_likelihood_change)
    if log_likelihood_change < 0.01:
        return 1
    elif num_iterations >= 10:
        return 1
    else:
        return 0

known_training_seq = Trainer.TrainingSequence(rolls, states)

trainer = Trainer.KnownStateTrainer(standard_mm)
trained_mm = trainer.train([known_training_seq])

if VERBOSE:
    print(trained_mm.transition_prob)
    print(trained_mm.emission_prob)

test_rolls, test_states = generate_rolls(300)

predicted_states, prob = trained_mm.viterbi(test_rolls, DiceTypeAlphabet())
if VERBOSE:
    print("Probabilitas: %f" % prob)
    Utilities.pretty_print_prediction(test_rolls, test_states, predicted_states)

training_seq = Trainer.TrainingSequence(rolls, Seq("", DiceTypeAlphabet()))

trainer = Trainer.BaumWelchTrainer(baum_welch_mm)
trained_mm = trainer.train([training_seq], stop_training)

if VERBOSE:
    print(trained_mm.transition_prob)
    print(trained_mm.emission_prob)

test_rolls, test_states = generate_rolls(300)

predicted_states, prob = trained_mm.viterbi(test_rolls, DiceTypeAlphabet())

print("Prediction probability: %f" % prob)
Utilities.pretty_print_prediction(test_rolls, test_states, predicted_states)

