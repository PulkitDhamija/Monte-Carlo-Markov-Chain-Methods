%% setting parameters
n = 25000;
n0 = 5000;
T = readtable('Mroz Data.xlsx');
I = zeros(753,1);
I(:,1) = 1;
Tmat = T{:,:};
global X;
X = [I, Tmat(:,3), Tmat(:,4), Tmat(:,5), Tmat(:,6)];
global Y;
Y = Tmat(:,2);

% MCMC sampling
k = 5;
beta = zeros(k,1);
betas = zeros(k, 20000);
sig2 = 1;
sig2s = zeros(1, 20000);
z = zeros(753, 1);
for i = 1:753
    if Y(i) > 0 
        z(i) = Y(i);
    else
        u = unifrnd(0,1);
        Fc2 = normcdf(0,(X(i,:))*beta,sig2);
        z(i) = norminv(u*Fc2,(X(i,:))*beta,sig2);
    end
end

hw = waitbar(0,'Running...');
for i = 1:n
    sig2 = gensig2(beta, z);
    beta = genbeta(sig2, z);
    ssig2 = sqrt(sig2);
    for j = 1:753
        if Y(j) > 0 
            z(j) = Y(j);
        else
            u = unifrnd(0,1);
            Fc2 = normcdf(0,(X(j,:))*beta,ssig2);
            z(j) = norminv(u*Fc2,(X(j,:))*beta,ssig2);
        end
    end
    if(i > n0)
        betas(:,i-n0) = beta;
        sig2s(:,i-n0) = sig2;
    end
    if mod(i,floor(n/10))<1e-2
         waitbar(i/n,hw);
    end
end
close(hw)

disp(mean(sig2s));
disp(mean((betas')));

function val = genbeta(sig2, Y)
    k = 5;
    B0 = 1000.*eye(k);
    beta0 = zeros(k, 1);
    global X;
    invssig = sig2^(-1);
    B1 = inv( invssig.*((X')*(X)) + inv(B0) );
    betabar = (B1) * ( invssig.*((X')*(Y))  + (inv(B0))*beta0 );   
    val = (mvnrnd(betabar, B1))';
end

function val = gensig2(beta, Y)
    n = 753;
    global X;
    alpha0 = 100000;
    delta0 = 10;
    alpha1 = alpha0 + n;
    delta1 = delta0 + ((Y - X*beta)')*(Y - X*beta);
    draw = gamrnd(alpha1/2, 1./(delta1/2));
    val = 1./draw;
end