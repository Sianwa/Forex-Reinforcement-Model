from agent import AI_Trader
from functions import *
from tqdm import tqdm
import sys

# hyprtparameters
window_size = 10
episodes = 50

batch_size = 32
data = dataset_loader("AAPL")
data_samples = len(data) - 1
trader = AI_Trader(window_size)

# defining a training loop that will iterate through all the episodes
for episode in range(1, episodes + 1):
    print("Episode: {}/{}".format(episode, episodes))

    state = state_creator(data, 0, window_size + 1)

    total_profit = 0
    trader.inventory = []

    for t in tqdm(range(data_samples)):
        action = trader.act(state)

        next_state = state_creator(data, t+1, window_size + 1)
        reward = 0

        if action == 1:
            trader.inventory.append(data[t])
            print("AI Trader BOUGHT:", formatPrice(data[t]))

        elif action == 2 and len(trader.inventory) > 0:
            buy_price = trader.inventory.pop(0)
            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price
            print("AI Trader SOLD: ", formatPrice(
                data[t]), " Profit: " + formatPrice(data[t] - buy_price))
                
        if t == data_samples - 1 :
        # this for loop basically closes all open positions as the episode comes to an end. 
    
            for x in trader.inventory:
                buy_price = trader.inventory.pop(0)
                reward = max(data[t] - buy_price, 0)
                total_profit += data[t] - buy_price
                print("AI Trader SOLD: ", formatPrice(
                data[t]), " Profit: " + formatPrice(data[t] - buy_price))

            done = True

        else:
            done = False

        trader.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("Final Inventory", trader.inventory)
            print("########################")

           
        if len(trader.memory) > batch_size:
            print("USING MEMORY TO TRADE")
            trader.expReplay(batch_size)

   # save model weights and parameters at epochs divisible by 10
    if episode % 10 == 0:
        trader.model.save("models/ai_trader_{}.h5".format(episode))
