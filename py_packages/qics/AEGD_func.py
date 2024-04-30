import numpy as np

######################################################
# modules
######################################################

class AEGD():
    def est_GD(self, X, Y1, Y2, lr=1e-3, tol=1e-6, num_itr=100):
        """
        Estimation via Gradient Descent
        :param X: X
        :param Y1: Y1
        :param Y2: Y2
        :param lr: lr
        :param tol: tol
        :param num_itr: num_itr
        :return: A1 B A2 A
        """
        rr = Y1.shape[1]
        qq = Y2.shape[1]
        qq += rr

        # Step 0
        # Step 0.1: Set B=0, estimate A1
        A1_tmp, _, _, _ = np.linalg.lstsq(X, Y1, rcond=None)

        # Step 0.2: estimate B given A1_tmp
        B_tmp = self.est_B(A1_tmp, X, Y2)

        # Step m, m=1,2,...,num.itr
        for _ in range(int(num_itr)):

            # Step m.1: estimate A1_m given B_{m-1} via GD
            A1_new, _ = self.est_A1(B_tmp, A1_tmp, X,
                                    Y1, Y2, lr, 
                                    tol, num_itr=50)

            # Step m.2: estimate B given A1_m
            B_new = self.est_B(A1_new, X, Y2)

            norm_A = np.linalg.norm(A1_new - A1_tmp, 'fro')
            norm_B = np.linalg.norm(B_new - B_tmp, 'fro')
            norms = norm_A + norm_B
            # Update A1_m and B_m
            A1_tmp = A1_new
            B_tmp = B_new

            if norms <= tol:
                break

        A2_tmp = A1_tmp @ B_tmp
        A_tmp = np.concatenate((A1_tmp, A2_tmp), axis=1)
        return A1_tmp, B_tmp, A2_tmp, A_tmp

    def est_B(self, A1, X, Y2):
        """
        estimate B given A1
        :param A1: A1
        :param X: X
        :param Y2: Y2
        :return: a new B
        """
        X_new = X @ A1
        B_new, _, _, _ = np.linalg.lstsq(X_new, Y2, rcond=None)
        return B_new


    def est_A1(self, Bm, A1old, X, Y1, Y2, lr=1e-3, tol=1e-6, num_itr=100):
        """
        estimate A1 given B via gradient descent (GD)
        :param Bm: B estimate at this round
        :param A1old: A1 estimate last round
        :param X: X
        :param Y1: Y1
        :param Y2: Y2
        :param lr: lr
        :param tol: tol
        :param num_itr: num_itr
        :return: a new A1 estimate
        """

        A1_tmp = A1old
        pp, rr = A1_tmp.shape
        nn, qq = Y2.shape
        qq += rr

        for itr in range(int(num_itr)):
            # gradient of L_m^1, a pxr matrix
            grad1 = -2. * np.matmul(X.T, Y1 - X @ A1_tmp)

            grad2 = np.zeros((pp, rr))

            for i in range(nn):
                for j in range(qq-rr):
                    grad_XAB = X[i, :] @ A1_tmp @ Bm[:, j]
                    outerXB = np.outer(X[i, :], Bm[:, j])
                    grad2 += (Y2[i, j] - grad_XAB)*(-2)*outerXB

            ######################################
            # decaying learning rate
            lr *= 0.85**itr * 0.5**rr
            ######################################
            # update A1_tmp
            delta_tmp = lr*(grad1 + grad2)
            A1_tmp -= delta_tmp

            if np.linalg.norm(delta_tmp, 'fro') <= tol:
                break
        return A1_tmp, itr

######################################################
# functions
######################################################
def obj_val(E_est, rr, const_qq=2):
    """
    nn * log{| hat{E}^T hat{E} | } +  rr * ( log_2{nn} + const_qq * log_2{qq} )
    :param E_est: E_est
    :param rr: rr
    :param const_qq: const_qq
    :return: a list of three values: objective function, loss function, penalty function
    """

    nn, qq = E_est.shape
    Est2 = E_est.T @ E_est
    loss = nn*np.log(np.linalg.det(Est2))
    penalty = rr*(np.log2(nn) + const_qq*np.log2(qq))
    obj_value = loss + penalty
    return obj_value, loss, penalty


def loss2obj(loss_list, supList, nn, qq, const_qq=2):
    """
    covert a list of losses to its objective values by manually tuning the penalty term

    Args:
        loss_list (list): list of losses
        supList (list): support indices of losses
        nn (int): sample size
        qq (int): number of features
        const_qq (int, optional): a qq-related constant. Defaults to 2.

    Returns:
        obj_list (list): tuned objective values
    """

    obj_list = []

    for iloss, isup in zip(loss_list, supList):
        rr = len(isup)
        penalty = rr*(np.log2(nn) + const_qq*np.log2(qq))
        obj_list.append(iloss + penalty)
       
    return obj_list

def norm_scaled(mat_est, mat_true):
    """
    calculate scaled frobenius norm

    Args:
        mat_est (numpy.darray): estimate of a matrix
        mat_true (numpy.darray): true value of a matrix

    Returns:
        float: scaled frobenius norm
    """
    nrow, ncol = mat_est.shape
    val = np.linalg.norm(mat_est - mat_true, 'fro') / np.sqrt(nrow*ncol)
    return val

