using Statistics
w = 5;
x = 7000;
y = x;
I = rand(x,y)
F = zeros(x,y)
# now using 1 core [time = 15s for x = 5000]
ww = floor(Int,w/2)
@time for i = (ww+1):1:(x-ww-1), j = (ww+1):1:(x-ww-1)
    chip = I[(i-ww):(i+ww), (j-ww):(j+ww)];
    F[i,j] = (I[i,j] - mean(chip))/std(chip[:])
end
# now using 4 threads [time = 7s for x = 5000]
@time Threads.@threads for i = (ww+1):1:(x-ww-1)
    for j = (ww+1):1:(x-ww-1)
    chip = I[(i-ww):(i+ww), (j-ww):(j+ww)];
    F[i,j] = (I[i,j] - mean(chip))/std(chip[:])
    end
end