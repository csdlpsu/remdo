function f = mystery(x, y)
    % An arbitrary nonconvex function, evaluated in MATLAB.
    % Smooth quadratic bowl centered near (2, -1), plus a sinusoidal
    % ripple and a little x-y coupling to make it nontrivial.
    f = (x - 2).^2 + 3*(y + 1).^2 + sin(2*x).*cos(2*y) + 0.3*x.*y;
end
