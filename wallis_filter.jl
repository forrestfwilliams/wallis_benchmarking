using Statistics
using ImageFiltering

# wrapping in a function call to avoid compile time in metrics
function filter_test()
w = 5;
x = 7000;
y = x;
I = rand(x,y)
F = zeros(x,y)

# define wallis filter
function wallis(chip)
    chip_mean = mean(chip)
    kernel_sum = length(chip)
    chip_std = sqrt(sum((chip .- chip_mean).^2)/kernel_sum)
    return out = (chip[ceil(Int,kernel_sum/2)] - chip_mean) / chip_std
end

ww = floor(Int,w/2)

# now using 1 core [time = 27s for x = 7000]
@time for i = (ww+1):1:(x-ww-1), j = (ww+1):1:(x-ww-1)
    F[i,j] = wallis(I[(i-ww):(i+ww), (j-ww):(j+ww)])
end

# now using mutiplt threads (if Julia setup to use multiple threads) [time = 11s for x = 7000]
@time Threads.@threads for i = (ww+1):1:(x-ww-1)
    for j = (ww+1):1:(x-ww-1)
        F[i,j] = wallis(I[(i-ww):(i+ww), (j-ww):(j+ww)])
    end
end

# now using mapwindow and mutiplt threads (if Julia setup to use multiple threads) [time = 9s for x = 7000]
@time F = mapwindow(wallis, I, (w,w));

# now try using imfilter [time = <1s for x = 7000]
@time begin
    Im = imfilter(I, ones(w,w)/w^2);
    Is = I - Im;
    Istd = sqrt.(imfilter(Is.^2, ones(w,w)/w^2))
    Is = Is ./ Istd;
end

end

# run test
filter_test();
