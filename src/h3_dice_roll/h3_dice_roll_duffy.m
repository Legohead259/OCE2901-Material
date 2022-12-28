% Assignment 3: Dice Roll
% Braidan Duffy
% Created: 01/22/2021
% Last modified: 12/28/2022

%% Part 1
% For this first part, we will write a script that simulates rolling a dice and performs a basic statical analysis and generates a bar plot. 
% We will roll the dice 10 times, then 1000 times and finish with an output table and figure that show all the relvant information

rolls_10, mean_10, std_10 = roll_die(10);        % Generate array of 10 dice rolls
rolls_1000, mean_1000, std_1000 = roll_die(1000);    % Generate array of 1000 dice rolls

% Generate histograms of results
figure(1)
hist_10 = histogram(rolls_10);
xlabel('Roll Value')
ylabel('Count')
title('Histogram of 10 Dice Rolls')
figure(2)
hist_1000 = histogram(rolls_1000);
xlabel('Roll Value')
ylabel('Count')
title('Histogram of 1000 Dice Rolls')

%% Question 2

rolls_two_10, mean_two_10, std_two_10  = roll_two_dice_sum(10);       % Generate array of 10 dice rolls
rolls_two_1000, mean_two_1000, std_two_1000 = roll_two_dice_sum(1000);   % Generate array of 1000 dice rolls

% Generate histograms of results
figure(3)
hist_two_10 = histogram(rolls_two_10);
xlabel('Roll Value')
ylabel('Count')
title('Histogram of Two Summed Dice Rolls (10x)')
figure(4)
hist_two_1000 = histogram(rolls_two_1000);
xlabel('Roll Value')
ylabel('Count')
title('Histogram of Two Summed Dice Rolls (1000x)')


%% Utility

% Rolls a singular dice a specified number of times
% @param itr: number of times dice is rolled
% @return an array of dice rolls
function [rolls, mean, std] = roll_die(itr)
    rolls = zeros(itr);
    for i = 1:itr
        _rolls[i] = randi(6);
    end
    _mean = mean(_rolls);
    _std = std(_rolls);
end

% Rolls two dice and sums their values a specified number of times
% @param itr: number of times dice is rolled
% @return an array of two summed dice rolls
function [_rolls, _mean, _std] = roll_two_dice_sum(itr)
    rolls = zeros(itr);
    for i = 1:itr
        _rolls[i] = randi(6) + randi(6);
    end
    _mean = mean(_rolls);
    _std = std(_rolls);
end
