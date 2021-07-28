library (philentropy);
library (utils);
library (mldr.datasets);
library (doMC);
library (bigmemory);
library (parallelDist);

source ("src/utils.R");

################

get_pij <- function (d)
{
  diag (d) <- Inf;
  return (exp (-d) / rowSums (exp (-d)));
}


get_pij_wts <- function (d, wts = NA)
{
  if (sum (is.na (wts)) > 0)
  {
    wts <- rep (1, nrow (d)); # Equal weights
  }

  diag (d) <- Inf;
  to_ret <- exp (-d) / rowSums (exp (-d));
  to_ret <- sweep (to_ret, 2, wts, "*");
  return  (to_ret / rowSums (to_ret));
}

get_batch_range <- function (idx_to_proc, mbatch, batch_nos)
{
  n <- length (idx_to_proc);
  block <- ceiling (n / batch_nos);
  start <- ((mbatch - 1) * block) + 1;
  end <- start + block - 1;
  if (end > n) { end <- n; }
  
  return (start:end);
}


nca <- function (mdata, Y = NULL, A = NULL, lambda = 0.1, gamma = 0.9, reg_factor = 0, max_epoch = 10, dim = ncol (X), cores = 8, batch_size = 32, xdist = "euclidean", ydist = "jaccard", debug_flag = FALSE)
{
  if (class (mdata) == "mldr")
  {
    xy <- mldr_to_xy (mdata);
    X <- as.matrix (xy$X);
    Y <- xy$Y;
  }
 
  
  doMC::registerDoMC (cores = cores);
  batch_nos <- cores;
  
  if ((batch_size == -1) || (batch_size > nrow (X)))
  {
    batch_size <- nrow (X);
    cat ("Setting batch size to full batch\n");
  }
  
  if (is.na (dim) || (dim == -1))
  {
    dim = ncol (X);
  }
  
  
  c <- ncol (X);
  r <- dim;
  if (is.null (A))
  {
    A <- matrix (rnorm (r * c), nrow = r, ncol = c) * 0.01;
    # TODO: Implement PCA initialisation, other initialisations.
  }
  
#   lab_sim <- 1 - philentropy::distance (Y, method = "jaccard");
  # TODO: can have weights per label here? 
  lab_sim <- 1 - as.matrix (parallelDist::parDist (as.matrix (Y), method = ydist, threads = cores));
  if (ydist == "hamming")
  {
    lab_sim <- 1 - lab_sim;
  }
#   print (lab_sim)
  
  # Explore thresholding instead of weighting small ones. 
  # NOTE ALSO: we can weight specific labels as well, find weighting and then multiply
  #   lab_sim[lab_sim > 2/ncol (Y)] <- 0;
  
  
  cost_list <- c ();
  
  debug_data <- list ();
#   debug_data$models <- list ();

  # TODO: Validate code, make sure it is implemented perfectly
  # optimise for space, (dmat, dmat_orig)
  # vectorise computation
  # see https://kevinzakka.github.io/2020/02/10/nca/#tricks
#   dmat_orig <- as.matrix (dist (X)); # Distance in original space
  velocity <- 0;
#   dmat_orig <- as.matrix (parallelDist::parDist (as.matrix (X), method = xdist, thread = cores)); # Distance in original space
  for (epoch in 1:max_epoch)
  {
    start_time <- Sys.time();
    if(debug_flag == TRUE)
    {
      debug_data$models[[epoch]] <- list (A = A);
      class (debug_data$models[[epoch]]) <- "nca_m";
    }
    
    X_trans <- t (A %*% t (X));
#     dmat <- as.matrix (dist (X_trans));
    dmat <- as.matrix (parallelDist::parDist (X_trans, method = xdist, thread = cores));
    pij_mat <- get_pij (dmat);
#     p_i <- rep (NA, nrow (X));
    p_i <- big.matrix  (nrow = nrow (X), ncol = 1); # shared memory

    idx_to_proc <- 1:nrow (X); # Full batch
    
    
    progress_arr <- bigmemory::big.matrix  (nrow  = cores, ncol = 1, type = "double");
    for (cnt in 1:cores) { progress_arr[cnt] <- 0; }
    pb <- txtProgressBar(min = 1, max = length (idx_to_proc), style = 3);
    full_grad <- foreach (mbatch = 1:batch_nos, .combine = "+") %dopar% 
    {
      grad <- matrix (0, nrow = ncol (X), ncol = ncol (X));
      this_idx_to_proc <- idx_to_proc[get_batch_range (idx_to_proc, mbatch, batch_nos)];

      for (i in this_idx_to_proc)
      {
        progress_arr[mbatch] <<- progress_arr[mbatch] + 1; # NOTE: Global update
        setTxtProgressBar(pb, sum (bigmemory::as.matrix (progress_arr)));
        ci_vec <- lab_sim[i,];
        this_pi <- sum (ci_vec * pij_mat[i,]);
        
        part1 <- part2 <- matrix (0, nrow = ncol (X), ncol = ncol (X));
        this_batch_idx <- 1:nrow (X);
        #this_batch_idx <- sample (1:nrow (X), batch_size);
        for (k in sample (this_batch_idx, batch_size)) # mini batch, use batch_size = nrow (X) for full batchs
        {
          term <- t (X[i,,drop=F] - X[k,,drop=F]) %*% (X[i,,drop=F] - X[k,,drop=F]);
#           term <- (X[i,] - X[k,]) %*% t (X[i,] - X[k,]);
#           print (dim (term));
          part1 <- part1 + term * pij_mat[i,k];
          part2 <- part2 + term * ci_vec[k] * pij_mat[i,k];
        }
        part1 <- this_pi * part1;
        grad  <- grad + (part1 - part2);
        
        p_i[i] <<- this_pi;
      }
      grad;
    }
    close (pb);
    
      

    full_grad <- 2 * A %*% full_grad;
    
    cost_list <- c (cost_list, sum (bigmemory::as.matrix (p_i), na.rm = TRUE));
    
    regularisation <- reg_factor * A;
    
    # Momentum
    velocity <- gamma * velocity + full_grad * lambda;
    A <- A + velocity - regularisation;

    end_time <- Sys.time();
    cat ("iter = ", epoch, " | ", "cost = ", cost_list[length (cost_list)], " | ", difftime (end_time, start_time, unit = "secs"), "secs", "\n");
  }
 
  if(debug_flag == TRUE)
  {
    debug_data$models[[epoch+1]] <- list (A = A);
    class (debug_data$models[[epoch+1]]) <- "nca_m";
  }
  
  to_ret <- list (A = A, cost = cost_list, debug_flag = debug_flag, debug_data = debug_data);
  class (to_ret) <- "nca_m";
  return (to_ret);
}


predict.nca_m <- function (nca_m, mdata)
{
  if (class (nca_m) != "nca_m")
  {
    return (NULL);
  }
  
  if (class (mdata) == "mldr")
  {
    xy <- mldr_to_xy (mdata);
    X <- xy$X;
    Y <- xy$Y;
  }
  else
  {
    X <- mdata;
  }
  
  X <- t (nca_m$A %*% t (as.matrix (X)));
  if (class (mdata) == "mldr")
  {
    mdata <- xy_to_mldr (X, Y);
  }
  else
  {
    mdata <- X;
  }
  return (mdata);
}
