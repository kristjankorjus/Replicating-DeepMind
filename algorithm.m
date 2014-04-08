%% Explanations of variables
% x_i - a screenshot
% a_i - an action
% s_t = {x_1, a_1, x_2, a_2, ..., x_t} - sequence of screenshots and actions
% phi() - pre-processing function
% phi_i - pre-processed sequence
% phi_t = {phi(x_1), a_1, phi(x_2), a2, ..., phi(x_t)}
% theta - the "brain", i.e. parameters of the neural network
% Q - calculates expected value of the move using NN
% Q* - ?what's the difference between Q* and Q?
% y_i - current + expected future rewards for changing theta with radient descent 
% gamma - discount factor

%% Initialisations
% initialise memory D with size N
% initialise random action-value function Q
% constant epsilon = 0.05
% constant gamma 


%% Algorithm
for game=1:M
	% initialise sequence and preprocessing
	% s1 = {x_1}
	% phi1 = phi(s1)
	
	% t=time in one game, T = last time point of the game
	for t=1:T
		if(rand() <= 0.05)
			% select random action a_t with probability 0.05
		else
			% select a_t = maxarg_a Q*(phi(s_t), a, theta)
		end;
		
		% execute a_t
		% observe reward r_t and screenshot x_(t+1)
		
		% set s_(t+1) = [s_t, a_t, x_(t+1)]   (add performed action and new screenshot to the memory of this game)
		% set phi_(t+1) = [phi, a_t, phi(x_(t+1)]   (add processed screenshot to processed memory of this game)
		
		% store transition (phi_t, a_t, r_t, phi_t+1) into D
		
		% take a random minibatch (let's take 10) of transitions from D
		for %each example in the minibatch
			if %phi_(j+1) is terminal (game is over)
				y_j = r_j;
			else
				y_j = r_j + gamma * max_a' Q(phi_(j+1), a', theta)% (y_j = r_j + max return from the next step)
			end;
		end;
		
		% Perform gradient descent step according to equation 3
		
	end;
end;
	
	
	
	