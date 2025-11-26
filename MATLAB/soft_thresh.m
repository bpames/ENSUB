function y = soft_thresh(a, t) 
   y = max(0, a - t) - max(0, -a - t); 
end