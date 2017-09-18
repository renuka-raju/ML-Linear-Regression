function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
   % inter=zeros(1,m);
    inter=(theta'*X')-y'; %1*m matrix ----->htheta(X)-y
    for i=1:size(theta,1)
    %mulv=zeros(1,m);
    %cost=zeros(m,1);
    mulv=X(:,i);          % select theta ith column---values--->X(i)
    %make it a matrix of 1*m---values--->X(i)j
    cost=mulv'.*inter;     % htheta(X(i))-y(i))*X(i)j
    theta(i)=theta(i)-((1/m)*(alpha*sum(cost)));
    end
   %theta
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
