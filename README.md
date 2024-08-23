{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9a124b3d-1dc2-4473-8ecf-ad04934d5800",
   "metadata": {},
   "source": [
    "# Introduction\n",
    "\n",
    "This repo contains code for implementing a 1D stochastic molecule simulation with discretized time in the gymnasium environment, allowing one to use reinforcement learning to investigate the properties of near-optimal controllers. It also contains several example programs used to investigate the utility of reinforcement learning in understanding control of stochastic molecular dynamics.\n",
    "\n",
    "It is known in the literature for stochastic molecular dynamics that [control schemes based on the entire history of molecular levels offer superior performance to controllers that only act based on instantaneous values](https://www.nature.com/articles/nature09333), but such control mechanisms are difficult to derive analytically. Here, we show the improvements available from sampling molecular histories instead of instantaneous values by using reinforcement learning on the problem of delayed molecular controllers, where we find that the variance compared to a lagg-free controller can be reduced by up to 14% by predicting based off the past 3 observations instead of only the most recent value.\n",
    "\n",
    "This is pictured in the following graph, showing that a control mechanism (learned via machine learning) with delays can more effectively approach the performance of the optimal no-delay controller by using histories of molecular trajectories."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cded678f-ae5a-4cb4-8216-dbdf63adb49c",
   "metadata": {},
   "source": [
    "![Comparison of Avg5 dt05](Images/Comparison_Of_Avg5_dt05.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "756f9a49-9a8d-4ade-a488-389b01778419",
   "metadata": {},
   "source": [
    "# How To Use\n",
    "\n",
    "molcontrol.py contains the following functions:\n",
    "\n",
    "- gymn environment for running a molecular simulation subject to stochastic death events\n",
    "- functions to create and train a PyTorch neural network with replay memory with this gym environment\n",
    "- functions to run the analytically computed optimal controller for this gym environment\n",
    "- functions to train a lookup table on the gym environment for arbitrary training data (allowing for delayed controllers that use histories)\n",
    "- some programs to visualize the results of the above training\n",
    "\n",
    "In the folders contain examples of using this code to analyze various questions related to controlling the variability of singular molecules. The PyTorch functions are only used for the neural net which was not used for most of the analyses done, so if there are issues installing PyTorch it can be removed from the molcontrol.py file without any issues.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d795358f-b0a0-4493-bc97-e286ccadd969",
   "metadata": {},
   "source": [
    "# Purpose\n",
    "\n",
    "Reducing variability is a common goal in many control applications. In this program, we analyze the properties of near-optimal controllers for a discrete time process where a finite number of molecules randomly decay between timesteps with probability $p$. The controller only sees the number of molecules at the current timestep $N_k$ (or potentially a number of pevious timesteps $N_{k -1}, N_{k - 2}...$) and must decide upon the optimal amount of number of molecules to be produced in the next step to ensure that the observed number of molecules at the next timestep is as close to a target $B$ as possible, with the goal being the minimize the error, given by the squared differenece between the number of molecules and the target.\n",
    "\n",
    "This problem can be analytically solved in the case where the controller observes exactly the current number of molecules before making it's decision for the next timestep, where the optimal number of molecules to send in at timestep k, $A_k$ is given by the closest integer to\n",
    "\n",
    "\\begin{equation}\n",
    "A_{k, cont.} = B - N_{k}(1 - p)\n",
    "\\end{equation}\n",
    "\n",
    "obtained by minimizing expected value of the squared difference. This result implies the number of molecules made should compensate for the average number of molecules lost. Unfortunately, analytically solving for the average error is difficult, and one must use numerical simulations. For the details of this solution, see the folder \"Theoretical Results\".\n",
    "\n",
    "Because this system is a markov chain, there is no increase in predicting future values of molecules given we know the current number of molecules. However, if our controller does not observe the most recent value of the abundance, but instead a lagged history of molecular abundances (while simultaneously receiving error estimates based on the current but invisible value), this markov property does not hold, and observing large histories of the current molecule may result in increased performance. \n",
    "\n",
    "Such controllers are difficult to analytically solve for. Instead, we implement a reinforcement learning algorithm to learn a near-optimal controller for us.\n",
    "\n",
    "## Near Optimal Controllers with Reinforcement Learning\n",
    "\n",
    "Reinforcement learning is a branch of machine learning concerned with teaching agents to solve problems by providing them with rewards and punishments based off performance. We will use it here to create a controller that will take in as input a history of the past molecular observations, and output the number of molecules it believes should be added into the system at this timestep. \n",
    "\n",
    "### Environment\n",
    "\n",
    "Reinforcement learning requires an environment - in this case, a simulation of the system to learn from. We implemented this system in pythons gymnasium library, a standard library for reinforcement learning algorithms. Our environment takes in \n",
    "\n",
    "- Initial number of molecules\n",
    "- Average molecule lifetime $t_{mol}$\n",
    "- dt, the time between observations\n",
    "- The maximum number of steps to run the simulation for\n",
    "- The length of history of the molecule's abundances\n",
    "- the target value\n",
    "- the maximum number of molecules allowed in the system\n",
    "\n",
    "which is used to compute the probability a molecule does not decay $p = exp(-dt/t_{mol})$. At every step, it takes in an action by the controller, and computes the number of molecules that have decayed since the last observation (drawn from a binomial with probability $p$ before adding in the action, and returning the history of the molecules abundances, the reward (negative squared difference between the current value and the target), and the actual optimal number of molecules to send in, along with some generic flags for gymnasium libraries.\n",
    "\n",
    "### Neural Networks\n",
    "\n",
    "The network learns using the loss, given by the squared difference between the network's taken action and the optimal action. This network is capable of learning, but comparing it to the analytic optimal controller for this problem reveals that it consistently does worse than the optimal solution, and furthermore it's performance does not improve with more training. This is a common problem in the field of deep reinforcement learning, where increased amounts of training do not result in greater performance, as shown here by the departure of the neural net predictions from the optimal controller."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "16295798-fa6e-4625-90e3-fa80fcd1b369",
   "metadata": {},
   "source": [
    "![Neural_Nets_Get_Worse](Images/Neural_Nets_Get_Worse.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df8ea63a-077c-4af9-a58d-e7a52224857d",
   "metadata": {},
   "source": [
    "For details, see \"Neural Nets.ipynb\"\n",
    "\n",
    "### Tabular Methods\n",
    "\n",
    "An alternative to using neural networks is lookup tables. Unlike networks, they cannot infer general principles and solve problems that they have never observed and are only suitable for small action spaces due to memory constraints, but in exchange they never undergo common issues with deep learning RL approaches such as catastrophic forgetting. Since the state space of our problem is relatively small and our interest is in producing near-optimal controllers, they work well for this problem."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "45381592-f8f7-492b-9118-ba9732528961",
   "metadata": {},
   "source": [
    "![Lookup_Tables_get_better](Images/Lookup_Tables_get_better.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab2ea8b4-9fa9-48c7-be32-3d2d57f9a448",
   "metadata": {},
   "source": [
    "For details, see \"Lookup Tables.ipynb\".\n",
    "\n",
    "### Analyzing The Performance Of Our Machine Learning Model\n",
    "\n",
    "Learning curves are not effective measures of performance in this problem, because the systems learn the solution very quickly but are still subject to significant intrinsic noise. The significant intrinsic noise is much more notable than any increase from further training. Instead, to demonstrate our solutions performance, we run it for 100000 steps, and take the average of the last 50000 steps as the average error, and sample across multiple seeds to compare it's performance against the known optimal solution. We do this across a variety of averages and temporal resolutions. For the code, see the folder \"Analytic Model\"."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23c74404-e626-48b9-b75c-0994e8137167",
   "metadata": {},
   "source": [
    "#### Averages\n",
    "\n",
    "The machine learning solution and the analytic optimal solution match closely. The variability rises linearly as a function of the average. The results are run for 10 different seeds to produce points with error bars."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "36774c4c-bed9-44b8-960d-3f021cd654e0",
   "metadata": {},
   "source": [
    "![d05_comparison_analytic_model_net](Images/d05_comparison_analytic_model_net.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f4a77fa-c443-4185-9fc6-112802d35fa1",
   "metadata": {},
   "source": [
    "### Temporal Resolution\n",
    "\n",
    "Instead of comparing our analytic solution with our machine learning solution across a variety of averages, we can instead vary the time between timesteps (dt). We find again there is fairly good agreement between the two methods, except for an issue at $dt = 0.01$."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f56d667d-86df-4305-9e67-0a3c3718d784",
   "metadata": {},
   "source": [
    "![avg20_comparison_analytic_model](Images/avg20_comparison_analytic_model.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0757d151-9750-479f-9518-a9ac601fa76c",
   "metadata": {},
   "source": [
    "Note that variability is a nonlinear function of the temporal resolution. This is because very small and very small timesteps are easy to control: in the former, only one molecule decays at most, while in the latter, almost all molecules are guaranteed to have decayed. Ensuring that the target value is met is simple because the source of the randomness (decays) is very small.\n",
    "\n",
    "However, this large timestep limit is not very useful - it is an artifact of how we've chosen to implement our simulation, and the low variability in this case is not useful for actual control problems. Thus for further analysis, we stick to analyzing across a variety of averages with a modest timestep ($dt = 0.5$)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f7902a4-d559-4312-8e3c-0a0c4252c5fb",
   "metadata": {},
   "source": [
    "### Testing Markovianity\n",
    "\n",
    "The system being controlled is markovian (the history of molecule numbers offers no predictive advantage if the most recent molecule count is known). We can test to see if our controller can learn this, modifying our algorithm to keep track of and use a larger history of molecular values. This should produce the same performance as before (matching the optimal solution which only uses the latest value), which is what we observe. However, there is a slight performance loss at higher averages, which makes sense since our system will now learn much slower since it is tracking non-useful variables and higher averages take longer to learn.\n",
    "\n",
    "For the code, see folder \"Extra Histories\"."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "278b0e06-c327-44c9-a0f5-74c583d8d681",
   "metadata": {},
   "source": [
    "![d05_comparison_extra_history](Images/d05_comparison_extra_history.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bcce529d-5509-485e-8d84-6f2c3b1ceb57",
   "metadata": {},
   "source": [
    "We can also plot our controller's actions in 2D plot, which shows the machine learning controller only cares about the most recent molecular value when making predictions to minimize future variability. This is show by the constant values along the lagged axis."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5428ec2a-bd53-415c-bcbc-fa12e98b382b",
   "metadata": {},
   "source": [
    "![action_values_extra_history](Images/action_values_extra_history.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "663cdfdf-7c4f-48f2-97bf-cb40d0deb95e",
   "metadata": {},
   "source": [
    "Note that if the value of the molecule 1 timestep ago was very large, then theres a large chance the current value is large (reflected by the large bottom left corner of learned values)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ee87385-1d0b-4921-8395-a9cc9f1fe7f3",
   "metadata": {},
   "source": [
    "## Non-Analytic Control Problems\n",
    "\n",
    "Having shown our machine learning algorithm can match our optimal analytic controller for simple problems, we apply it to a complex case where analytic controllers are difficult to solve for. For the code, see folder \"Non-Markovian Models\".\n",
    "\n",
    "### Lagged Controllers\n",
    "\n",
    "The case studied here is a delayed controller: it's actions will not be enacted at the next timestep (as was done previously) but instead is performed several timesteps later. This case is studied in the folder \"Non-Markovian Models\" because the system is no longer Markovian in the variables seen by the controller, and the controller does notably worse compared to the instantaneous controller."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e954242-c48a-423e-bfbf-b5e460a6f8c7",
   "metadata": {},
   "source": [
    "![Lagg_Makes_Worse_Controllers_dt05](Images/Lagg_Makes_Worse_Controllers_dt05.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "868c64f2-9c86-416e-8e5b-8b8489554e23",
   "metadata": {},
   "source": [
    "Note that more delay (lagg) results in worse performance in controlling the variability. The blue and red curves are obtained by running our machine learning algorithm using most recent observation available to the controller."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34acb5a1-e5de-427b-83dc-555d9f112a4c",
   "metadata": {},
   "source": [
    "### History Dependent Machine Learning Controllers Can Improve Performance\n",
    "\n",
    "By running our machine learning algorithm on this model and allowing it to use information from older observations (not just the most recent observation) we can improve the performance of our algorithm. Intuitively, this is because by knowing the history of values observed by the controller, it can estimate to some degree the inputs the molecule will see in the future (beyond the current value observed by the controller) and the variability lies only in the number of deaths and not in the number of molecular production events.\n",
    "\n",
    "This is most apparent if we plot the optimal actions of our controller in the case where it observes the two most recent values, which now show a clear dependence on older observations compared to the simple markovian model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7a5d09e-ee97-4a37-8122-2df4b6a83dd8",
   "metadata": {},
   "source": [
    "![action_values_lagg1_history1](Images/action_values_lagg1_history1.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ffe67dc2-f91c-433c-bb00-0594c71d31ab",
   "metadata": {},
   "source": [
    "Although the change in controller behaviour is striking, the performance gain is less obvious."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e5789f15-c554-4767-8104-4db6d8eb6bf7",
   "metadata": {},
   "source": [
    "![Lagg1_With_History_dt05](Images/Lagg1_With_History_dt05.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df3c85dd-d4fb-4474-86dd-6ace2acea927",
   "metadata": {},
   "source": [
    "The controllers using additional observations (History 1, 2, and 1 2) perform somewhat better than the controller trained on only the latest observation, but the more complex controllers do worse at larger averages, suggesting that they require more training time (because of the larger state space for larger averages and complex controllers). We can more accurately evaluate the performance of the controllers by comparing the results at a small average, such as an average of 5:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2b1af99-cca1-4840-9f7e-a7b0460764f6",
   "metadata": {},
   "source": [
    "![Comparison_Of_Avg5_dt05](Images/Comparison_Of_Avg5_dt05.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a720c294-971d-45cb-9429-029dadfea7a4",
   "metadata": {},
   "source": [
    "### Numerical Results\n",
    "\n",
    "The effect of a history dependent controller reduced the variability from  1.82  to  1.752. The minimal variability for this average and temporal resolution with a controller without any delay is 1.28. The effect of using histories in our machine learning algorithm has reduced the error from this optimal performance from 0.42 to 0.37, resulting in a 14 percent improvement from using histories."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bdc588a8-9d31-4c43-b86d-5d6c9d079ce4",
   "metadata": {},
   "source": [
    "In general, investigating other average lengths results in a performance of slightly less, around 5% (which can be attributed to the longer training times needed for our controller to learn more complex control schemes)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be4e6d2a-aabe-4408-b4f8-e06fd6bf8f97",
   "metadata": {},
   "source": [
    "# Future Steps"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e49002ed-5df3-4e67-8dc4-09b34b9e318b",
   "metadata": {},
   "source": [
    "Future directions to take this repo are:\n",
    "\n",
    "- Running simulations with larger training times, to see if the increase in performance is consistent across all averages\n",
    "- Comparing lookup tables with deep learning approaches which can generalize and may learn more complex controllers faster\n",
    "- More complicated gym environments to investigate more interesting questions (eg, given that we control a molecule using only the abundance of a downstream molecule, what is the optimal lifetime ratio of the downstream molecule relative to the fluctuations of the controlled molecule to make the best controllers? This may have biological implications for the optimal lifetime of reporter molecules.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "205ca6e9-58c5-4e1b-b415-fd96f904045a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
