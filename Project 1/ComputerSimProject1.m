% CSCI-UA 330 Project 1
% Asteroid deflection with symplectic Euler + bisection for minimum dv
% Andre Llaneta - wpl5304

clear; clc; close all

% constants
G  = 6.67430e-11; % gravitational constant (m^3 kg^-1 s^-2)
M  = 5.97219e24; % mass of Earth (kg)
RE = 6.371e6; % radius of Earth (m)

% inputs
m_ast     = input("Asteroid mass m (kg) = ");
v0        = input("Initial speed (m/s) (don't choose anything <=0)= ");
b         = input("Y-axis starting offset parameter b (m) = ");
h         = input("Safety buffer h (m) (Earth Athmosphere = 1e5) = ");
rk        = input("Kick radius rk (m) (must satisfy RE < rk < r0) = ");
dt        = input("Time step dt (s) [shoule be between 10 and 200 (lower -> accurate, higher -> faster)] = ");
useRocket = input("Rocket equation? (1=yes,0=no) = ");

if useRocket==1
    Ufrel = input("U_frel (m/s) (Typically 2000-4500 for standard chemical engines) = "); % velocity of expelled propellant (higher -> more efficient, but more expensive) For chemical rockets, typically between 2000 and 4500 m/s
else
    Ufrel = NaN;
end

% setup and initial conditions
r0    = 1e11;         % start distance (m) (far enough to be effectively at infinity, but not too far to cause numerical issues)
Rstop = RE;           % impact stop radius (stops when it hits the Earth)
Rsafe = RE + h;       % miss criterion (earth + a safety buffer)
Rdone = r0;       % stop when outbound and far from Earth (for better looking plot)

if ~(rk > Rstop && rk < r0)
    % rk must be between Earth and the initial radius so the kick is applied on the modeled inbound trajectory
    error("Choose a different rk: must satisfy RE < rk < r0. Your r0 is %g m", r0) 
end

tmax = 20*r0/max(v0,1); % max time (s) (20 chosen as safety factor to allow for some orbits, but not too long to cause numerical issues)
clockmax = ceil(tmax/dt); % rounds up as tmax might not be an exact multiple of dt

% start with baseline of no deflection, then increase dv by bisection until it just misses (rmin >= Rsafe)
dv = 0; % change of velocity at rk (m/s)

% run baseline to get rmin without deflection, and to store the trajectory for plotting
[xsave0, ysave0, rmin0, xK0, yK0] = runtraj(dv, G, M, r0, b, v0, rk, dt, clockmax, Rstop, Rdone);

fprintf("\nBaseline closest approach rmin = %.6e m\n", rmin0)

% if already misses, no need to deflect
if rmin0 >= Rsafe
    fprintf("Already misses (>= RE+h). Required dv = 0\n");
    dv_best = 0;
    Eb = 0;
    mprop = 0;
    xsave = xsave0; ysave = ysave0;
    xK = NaN; yK = NaN;
    rmin_def = rmin0;
else
    % bracket dv so f(dv)=rmin(dv)-Rsafe crosses 0
    dv_lo = 0;
    dv_hi = 1e-6; % starts small to avoid overshooting, grows fast so it converges quickly

    for i = 1:50 % max 50 iterations to find upper bound but should converge much faster than that
        [~, ~, rmin_hi] = runtraj(dv_hi, G, M, r0, b, v0, rk, dt, clockmax, Rstop, Rdone);
        f_hi = rmin_hi - Rsafe;

        if f_hi >= 0 % found an upper bound that misses
            break
        end

        dv_hi = 2*dv_hi; % doubles dv_hi if it still hits

        if dv_hi > 1e6 % avoids infinite loop if dt too large or rk too small
            error("Couldn't bracket dv. Try smaller dt, bigger b, or earlier rk.")
        end
    end

    [~, ~, rmin_hi] = runtraj(dv_hi, G, M, r0, b, v0, rk, dt, clockmax, Rstop, Rdone);
    f_hi = rmin_hi - Rsafe;

    % checks if we found an upper bound that misses after 50 iterations
    if f_hi < 0
        error("Couldn't bracket dv. Try smaller dt, bigger b, or earlier rk.")
    end

    % bisection to find minimum dv that achieves f(dv)=rmin(dv)-Rsafe >= 0
    for i = 1:50
        dv_mid = 0.5*(dv_lo + dv_hi); % finds midpoint dv between current bounds

        [~, ~, rmin_mid] = runtraj(dv_mid, G, M, r0, b, v0, rk, dt, clockmax, Rstop, Rdone);
        f_mid = rmin_mid - Rsafe;

        % changes upper or lower bound depending on if dv_mid achieves a miss or not
        if f_mid >= 0
            dv_hi = dv_mid;
        else
            dv_lo = dv_mid;
        end

        % checks if the bounds are close enough to stop
        if (dv_hi - dv_lo) <= 1e-6*dv_hi
            break
        end
    end

    dv_best = dv_hi;

    % final deflected run to store trajectory + kick point + energy diagnostics
    [xsave, ysave, rmin_def, xK, yK, dE_kick_expected, dE_kick_sim, eps_post, eps_end] = runtraj(dv_best, G, M, r0, b, v0, rk, dt, clockmax, Rstop, Rdone);

    % finds kinetic energy of the deflection in J
    Eb = 0.5*m_ast*dv_best^2;

    if useRocket==1
        % rocket equation: m(t2)/m(t1) = exp(-||U2-U1||/||U_frel||)
        % ||U2-U1|| = dv_best, and m(t2)=m_ast
        m2 = m_ast;
        m1 = m2*exp(dv_best/Ufrel);
        mprop = m1 - m2;
        mr = m2/m1;
    else
        mprop = NaN;
        mr = NaN;
    end

    % print results
    fprintf("\nKick radius rk = %.6e m\n", rk)
    fprintf("Required dv = %.9f m/s\n", dv_best)
    fprintf("Energy = %.6e J\n", Eb)
    fprintf("Closest approach after kick rmin = %.6e m\n", rmin_def)

    % verify kick energy change and post-kick energy drift
    rel_tol_kick = 1e-6;    % tolerance for kick energy error (should be very small since it's not numerical)
    rel_tol_drift = 1e-3;   % tolerance for post-kick energy drift (larger since it's numerical)
    rel_kick_err = abs(dE_kick_sim - dE_kick_expected)/max(1e-12, abs(dE_kick_expected)); % compares simulated and expected specific energy change from the kick
    rel_drift = abs(eps_end - eps_post)/max(1e-12, abs(eps_post)); % compares specific orbital energy after kick and at ends of simulation to check for energy drift due to numerical errors

    fprintf("Post-kick specific-energy diagnostics:\n")
    fprintf("  Simulated dE_kick = %.6e J/kg\n", dE_kick_sim)
    fprintf("  Expected dE_kick  = %.6e J/kg\n", dE_kick_expected)
    fprintf("  Kick dE rel error = %.6e (tol = %.1e)\n", rel_kick_err, rel_tol_kick)

    % checks if the kick energy error is within tolerance
    if rel_kick_err <= rel_tol_kick
        fprintf("  Kick energy check = PASS\n")
    else
        fprintf("  Kick energy check = FAIL\n")
    end

    % checks if the post kick energy drift is within tolerance
    fprintf("  Post-kick drift   = %.6e (tol = %.1e)\n", rel_drift, rel_tol_drift)
    if rel_drift <= rel_tol_drift
        fprintf("  Drift check       = PASS\n")
    else
        fprintf("  Drift check       = FAIL (try smaller dt)\n")
    end

    % print rocket equation results
    if useRocket==1
        fprintf("Propellant mass = %.6e kg\n", mprop)
        fprintf("m(t2)/m(t1) = %.6e\n", mr)
    end
end

% plots
th = linspace(0,2*pi,400);
xe = RE*cos(th);   ye = RE*sin(th);
xs = Rsafe*cos(th); ys = Rsafe*sin(th);

% Earth view
% radii along each saved trajectory
r0_traj  = hypot(xsave0, ysave0);
r1_traj  = hypot(xsave,  ysave);

% indices of closest approach
[~, i0] = min(r0_traj);      % no kick closest approach index
[~, i1] = min(r1_traj);      % with kick closest approach index

% center on deflected closest approach
xc = xsave(i1);
yc = ysave(i1);

figure; hold on; grid on; axis equal

% draw Earth + safety buffer
plot(xe, ye, 'k-',  'LineWidth', 1.8)
plot(xs, ys, 'k--', 'LineWidth', 1.2)

% plot full trajectories
plot(xsave0, ysave0, 'LineWidth', 1.2)
plot(xsave,  ysave,  'LineWidth', 1.6)

% mark closest-approach points
plot(xsave0(i0), ysave0(i0), 'o', 'MarkerSize', 7, 'LineWidth', 1.6)
plot(xsave(i1),  ysave(i1),  'o', 'MarkerSize', 7, 'LineWidth', 1.6)

% mark kick if present
if ~isnan(xK)
    plot(xK, yK, 'ko', 'MarkerSize', 8, 'LineWidth', 1.8)
end

title("Earth View")
xlabel("x (m)"); ylabel("y (m)")

zoomRE = 5; % zoom in around Earth to show closest approach

xlim(xc + zoomRE*RE*[-1 1]);
ylim(yc + zoomRE*RE*[-1 1]);

legend("Earth (RE)", "Safety (RE+h)", "No kick", "With kick", ...
       "No-kick closest pt", "Kick closest pt", "Kick", ...
       'Location','best')

% Kick view
figure; hold on; grid on; axis equal

% draw Earth + safety buffer
plot(xe, ye, 'k-',  'LineWidth', 1.8)
plot(xs, ys, 'k--', 'LineWidth', 1.2)

% plot full trajectories
plot(xsave0, ysave0, 'LineWidth', 1.2)
plot(xsave,  ysave,  'LineWidth', 1.6)

% mark kick if present
if ~isnan(xK)
    plot(xK, yK, 'ko', 'MarkerSize', 8, 'LineWidth', 1.8)
end
title("Kick View")
xlabel("x (m)"); ylabel("y (m)")
if ~isnan(xK)
    c1 = 0.5*xK;    % centers the plot at half the kick radius
    c2 = 0.5*yK;
    span = 1.15*max(sqrt(xK^2+yK^2), 50*RE); % sets the span to be slightly larger than the kick radius, but at least 50 RE so it's not too zoomed in
    xlim(c1 + span*[-1 1]);     % sets x limits centered at c1 with the calculated span
    ylim(c2 + span*[-1 1]);     % sets y limits centered at c2 with the calculated span
else
    zoomKick = 3000;            % if no kick, just zoom in around the Earth to show the trajectories
    xlim(zoomKick*RE*[-1 1]);
    ylim(zoomKick*RE*[-1 1]);
end
legend("Earth (RE)", "Safety (RE+h)", "No kick", "With kick", "Kick", 'Location', 'best')



function [xsave, ysave, rmin, xK, yK, dE_kick_expected, dE_kick_sim, eps_post, eps_end] = runtraj(dv, G, M, r0, b, v0, rk, dt, clockmax, Rstop, Rdone)

    % initial conditions
    x = -r0;    % assumes asteroid starts on the left and moves to the right
    y =  b;     % Y-axis starting offset parameter
    u =  v0;    % initial velocity
    v =  0;     % no initial vertical velocity, just horizontal

    % creates an array of zeroes to store the trajectory for plotting
    xsave = zeros(1,clockmax);
    ysave = zeros(1,clockmax);

    rmin = 1e99; % starts high so any approach will be smaller, updated every step to find closest approach

    % initializes variables to store the kick
    kicked = 0;
    xK = NaN; yK = NaN;
    dE_kick_expected = NaN;
    dE_kick_sim = NaN;
    eps_post = NaN;
    eps_end = NaN;

    turned = 0;                 % becomes 1 after it misses
    r_prev = sqrt(x^2 + y^2);   % stores the previous radius

    for clock = 1:clockmax

        % finds current radius and updates minimum radius
        r = sqrt(x^2 + y^2);
        if r < rmin
            rmin = r;
        end

        % stop if impact
        if r <= Rstop
            xsave = xsave(1:clock-1);
            ysave = ysave(1:clock-1);
            eps_end = 0.5*(u^2 + v^2) - G*M/max(r, 1e-12);
            return
        end

        % radial velocity
        vr = (x*u + y*v)/max(r,1e-12);
        if vr > 0   % distance from earth increasing meaning asteroid is outbound
            turned = 1;
        end

        % stop if done outbound
        if turned==1 && r >= Rdone
            xsave = xsave(1:clock-1);
            ysave = ysave(1:clock-1);
            eps_end = 0.5*(u^2 + v^2) - G*M/max(r, 1e-12);
            return
        end

        % kick trigger: crosses rk (only once)
        if kicked==0 && (r_prev > rk) && (r <= rk)
            u_pre = u;
            v_pre = v;
            eps_pre = 0.5*(u_pre^2 + v_pre^2) - G*M/max(r, 1e-12); % specific orbital energy immediately before the kick
            s = sqrt(u^2 + v^2);    % speed magnitude
            nx = -v/s;              % unit normal vector perpendicular to velocity
            ny =  u/s;              % points to the left of the velocity vector

            rx = x/r;               % unit radial vector from Earth to asteroid
            ry = y/r;               % points from Earth to asteroid

            % applies the kick in the direction that increases the radius
            if (nx*rx + ny*ry) > 0  % dot product between normal and radial unit vector
                u = u + dv*nx;      % updates velocity with the kick
                v = v + dv*ny;      % updates velocity with the kick
                du = dv*nx;         % change in velocity components due to the kick
                dvv = dv*ny;        % used for energy change calculation
            else
                u = u - dv*nx;
                v = v - dv*ny;
                du = -dv*nx;
                dvv = -dv*ny;
            end

            eps_post = 0.5*(u^2 + v^2) - G*M/max(r, 1e-12); % specific orbital energy immediately after the kick
            dE_kick_sim = eps_post - eps_pre;               % simulated specific energy change from the kick
            dE_kick_expected = u_pre*du + v_pre*dvv + 0.5*(du^2 + dvv^2); % expected specific energy change from the kick
            kicked = 1;
            xK = x;
            yK = y;
        end

        r_prev = r;

        % symplectic Euler
        u = u - dt*G*M*x/(r^3);     % updates velocity with gravitational acceleration
        v = v - dt*G*M*y/(r^3);

        x = x + dt*u;
        y = y + dt*v;

        xsave(clock) = x;
        ysave(clock) = y;
    end

    % specific orbital energy at end of simulation
    r_end = sqrt(x^2 + y^2);
    eps_end = 0.5*(u^2 + v^2) - G*M/max(r_end, 1e-12);
end
