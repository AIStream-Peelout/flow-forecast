series_loss = 0.0
prior_loss = 0.0
for u in range(len(prior)):
    series_loss += (torch.mean(my_kl_loss(series[u], (
            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                    self.win_size)).detach())) + torch.mean(
        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size)).detach(),
                    series[u])))
    prior_loss += (torch.mean(my_kl_loss(
        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                self.win_size)),
        series[u].detach())) + torch.mean(
        my_kl_loss(series[u].detach(), (
                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        self.win_size)))))
series_loss = series_loss / len(prior)
prior_loss = prior_loss / len(prior)

rec_loss = self.criterion(output, input)

loss1_list.append((rec_loss - self.k * series_loss).item())
loss1 = rec_loss - self.k * series_loss
loss2 = rec_loss + self.k * prior_loss

if (i + 1) % 100 == 0:
    speed = (time.time() - time_now) / iter_count
    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    iter_count = 0
    time_now = time.time()

# Minimax strategy
loss1.backward(retain_graph=True)
loss2.backward()
self.optimizer.step()
