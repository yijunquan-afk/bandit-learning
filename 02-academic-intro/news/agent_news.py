import numpy as np
from base.agent import Agent
import numpy.linalg as npla
import scipy.linalg as spla

_SMALL_NUMBER = 1e-10
_MEDIUM_NUMBER = .01
_LARGE_NUMBER = 1e+2

class GreedyNewsRecommendation(Agent):
    """ 新闻推荐贪心算法 """

    def __init__(self, num_articles, dim, theta_mean=0, theta_std=1, epsilon=0.0,
                 alpha=0.2, beta=0.5, tol=0.0001):
        """初始化
        Args:
            num_articles (int): 新闻文章的数量  
            dim (int): 问题的维度数
            theta_mean (int, optional): θ分量的均值. Defaults to 0.
            theta_std (int, optional): θ分量的方差. Defaults to 1.
            epsilon (float, optional): 在epsilon-greedy算法中使用的参数. Defaults to 0.0.
            alpha (float, optional): 回溯直线搜索使用的参数. Defaults to 0.2.
            beta (float, optional): 回溯直线搜索使用的参数. Defaults to 0.5.
            tol (float, optional): 牛顿方法的停止标准. Defaults to 0.0001.
        """
        self.num_articles = num_articles
        self.dim = dim
        self.theta_mean = theta_mean
        self.theta_std = theta_std
        self.back_track_alpha = alpha
        self.back_track_beta = beta
        self.tol = tol
        self.epsilon = epsilon

        # 维护每篇新闻文章的最大后验估计与 Hessians矩阵
        self.current_map_estimates = [
            self.theta_mean*np.ones(self.dim) for _ in range(self.num_articles)]
        # np.diag: 构造对角线数组
        self.current_Hessians = [
            np.diag([1/self.theta_std**2]*self.dim) for _ in range(self.num_articles)]
        
        # 维护每篇文章的观测值
        self.num_plays = [0 for _ in range(self.num_articles)]
        self.contexts = [[] for _ in range(self.num_articles)]
        self.rewards = [[] for _ in range(self.num_articles)]


    def _compute_gradient_hessian_prior(self,x):
        """计算x处负对数似然先验部分的梯度和Hessian

        Arguments:
            x {_type_} -- _description_
        """ 
        Sinv = np.diag([1/self.theta_std**2]*self.dim)
        mu =  self.theta_mean*np.ones(self.dim)
        gradient = Sinv.dot(x-mu)
        Hessians = Sinv
        return gradient,Hessians


    def _compute_gradient_hessian(self, x, article):
        """computes gradient and Hessian of negative log-likelihood  
        at point x, based on the observed data for the given article."""

        g, H = self._compute_gradient_hessian_prior(x)

        for i in range(self.num_plays[article]):
            z = self.contexts[article][i]
            y = self.rewards[article][i]
            pred = 1/(1+np.exp(-x.dot(z)))

            g = g + (pred-y)*z
            H = H + pred*(1-pred)*np.outer(z, z)

        return g, H

    def _evaluate_log1pexp(self, x):
        """given the input x, returns log(1+exp(x))."""
        if x > _LARGE_NUMBER:
            return x
        else:
            return np.log(1+np.exp(x))

    def _evaluate_negative_log_prior(self, x):
        """returning negative log-prior evaluated at x."""
        Sinv = np.diag([1/self.theta_std**2]*self.dim)
        mu = self.theta_mean*np.ones(self.dim)

        return 0.5*(x-mu).T.dot(Sinv.dot(x-mu))

    def _evaluate_negative_log_posterior(self, x, article):
        """evaluate negative log-posterior for article at point x."""

        value = self._evaluate_negative_log_prior(x)
        for i in range(self.num_plays[article]):
            z = self.contexts[article][i]
            y = self.rewards[article][i]
            value = value + self._evaluate_log1pexp(x.dot(z)) - y*x.dot(z)

        return value

    def _back_track_search(self, x, g, dx, article):
        """Finding the right step size to be used in Newton's method.
        Inputs:
          x - current point
          g - gradient of the function at x
          dx - the descent direction

        Retruns:
          t - the step size"""

        step = 1
        current_function_value = self._evaluate_negative_log_posterior(
            x, article)
        difference = self._evaluate_negative_log_posterior(x + step*dx, article) - \
            (current_function_value + self.back_track_alpha*step*g.dot(dx))
        while difference > 0:
            step = self.back_track_beta * step
            difference = self._evaluate_negative_log_posterior(x + step*dx, article) - \
                (current_function_value + self.back_track_alpha*step*g.dot(dx))

        return step

    def _optimize_Newton_method(self, article):
        """Optimize negative log_posterior function via Newton's method for the
        given article."""

        x = self.current_map_estimates[article]
        error = self.tol + 1
        while error > self.tol:
            g, H = self._compute_gradient_hessian(x, article)
            delta_x = -npla.solve(H, g)
            step = self._back_track_search(x, g, delta_x, article)
            x = x + step * delta_x
            error = -g.dot(delta_x)

        # computing the gradient and hessian at final point
        g, H = self._compute_gradient_hessian(x, article)

        # updating current map and Hessian for this article
        self.current_map_estimates[article] = x
        self.current_Hessians[article] = H
        return x, H

    def update_observation(self, context, article, feedback):
        '''updates the observations for displayed article, given the context and 
        user's feedback. The new observations are saved in the history of the 
        displayed article and the current map estimate and Hessian of this 
        article are updated right away.

        Args:
          context - a list containing observed context vector for each article
          article - article which was recently shown
          feedback - user's response.
          '''
        self.num_plays[article] += 1
        self.contexts[article].append(context[article])
        self.rewards[article].append(feedback)

        # updating the map estimate and Hessian for displayed article
        _, __ = self._optimize_Newton_method(article)

    def _map_rewards(self, context):
        map_rewards = []
        for i in range(self.num_articles):
            x = context[i]
            theta = self.current_map_estimates[i]
            map_rewards.append(1/(1+np.exp(-theta.dot(x))))
        return map_rewards

    def pick_action(self, context):
        '''Greedy action based on map estimates.'''
        map_rewards = self._map_rewards(context)
        article = np.argmax(map_rewards)
        return article
##############################################################################
        
