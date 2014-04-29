%% Explanations of variables
% x_i - a screen image
% a_i - an action
% r_i - a reward
% s_t = {x_1, a_1, x_2, ..., a_{t-1}, x_t} - sequence of images and actions
% phi() - pre-processing function (smaller resolution, black and white, square)
% phi_t = {phi(x_1), a_1, phi(x_2), ..., phi(x_t)} - pre-processed sequence
% theta - the "brain", i.e. parameters of the neural network
% Q - calculates expected value of the action using NN with theta
% Q* - ?what's the difference between Q* and Q?
% y_i - current + expected future rewards for changing theta with radient descent 
% gamma - discount factor

%% Initialisations
% initialise memory D with size N
% initialise random action-value function Q (or may be its better to say that randomize theta, not Q)
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
		% observe reward r_t and image x_(t+1)
		
		% set s_(t+1) = [s_t, a_t, x_(t+1)]   (add performed action and new image to the memory of this game)
		% set phi_(t+1) = [phi, a_t, phi(x_(t+1)]   (add processed image to processed memory of this game)
		
		% store transition (phi_t, a_t, r_t, phi_t+1) into D
		
		% take a random minibatch (let's take 10) of transitions from D
		for %each example in the minibatch
			if %phi_(j+1) is terminal (game ended with the move)
				y_j = r_j;
        else
        % (y_j = r_j + max discounted return from the next step)
				y_j = r_j + gamma * max_a' Q(phi_(j+1), a', theta)
        end;

		
		% Perform gradient descent step according to equation 3
		
	end;
end;
	
	
	
	
