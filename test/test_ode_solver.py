import pytest
import torch

from RSB.sdes import SB_VESDE, SB_VPSDE


@pytest.mark.parametrize("sde_type", [SB_VESDE, SB_VPSDE])
def test_ode_step_matches_sb_se_table_two(sde_type) -> None:
    sde = sde_type(device="cpu")
    current = torch.tensor(0.8, dtype=torch.float64)
    next_time = torch.tensor(0.6, dtype=torch.float64)
    terminal = sde.T(current)
    state = torch.tensor([[[[0.2, -0.4]]]], dtype=torch.float64)
    prediction = torch.tensor([[[[0.5, 0.1]]]], dtype=torch.float64)
    observation = torch.tensor([[[[-0.3, 0.7]]]], dtype=torch.float64)

    alpha_current = sde.marginal_alpha(current)
    alpha_next = sde.marginal_alpha(next_time)
    alpha_terminal = sde.marginal_alpha(terminal)
    sigma_current = sde.marginal_sigma(current)
    sigma_next = sde.marginal_sigma(next_time)
    sigma_bar_current = sde.marginal_sigma_bar(current)
    sigma_bar_next = sde.marginal_sigma_bar(next_time)
    sigma_terminal_square = sde.marginal_sigma_square(terminal)
    weight_state = alpha_next * sigma_next * sigma_bar_next / (alpha_current * sigma_current * sigma_bar_current)
    weight_prediction = (
        alpha_next
        / sigma_terminal_square
        * (sigma_bar_next.square() - sigma_bar_current * sigma_next * sigma_bar_next / sigma_current)
    )
    weight_observation = (
        alpha_next
        / (alpha_terminal * sigma_terminal_square)
        * (sigma_next.square() - sigma_current * sigma_next * sigma_bar_next / sigma_bar_current)
    )
    expected = weight_state * state + weight_prediction * prediction + weight_observation * observation

    actual = sde.first_order_ode_sampling(
        x1=observation,
        xt=state,
        x0=prediction,
        t=current,
        t_prev=next_time,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("sde_type", [SB_VESDE, SB_VPSDE])
def test_ode_terminal_step_uses_finite_analytic_limit(sde_type) -> None:
    sde = sde_type(device="cpu")
    current = torch.tensor(1.0, dtype=torch.float64)
    next_time = torch.tensor(0.8, dtype=torch.float64)
    terminal = sde.T(current)
    prediction = torch.tensor([[[[0.5, 0.1]]]], dtype=torch.float64)
    observation = torch.tensor([[[[-0.3, 0.7]]]], dtype=torch.float64)
    weight_prediction = (
        sde.marginal_alpha(next_time) * sde.marginal_sigma_bar_square(next_time) / sde.marginal_sigma_square(terminal)
    )
    weight_observation = (
        sde.marginal_alpha(next_time)
        * sde.marginal_sigma_square(next_time)
        / (sde.marginal_alpha(terminal) * sde.marginal_sigma_square(terminal))
    )
    expected = weight_prediction * prediction + weight_observation * observation

    actual = sde.first_order_ode_sampling(
        x1=observation,
        xt=observation,
        x0=prediction,
        t=current,
        t_prev=next_time,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("sde_type", [SB_VESDE, SB_VPSDE])
@pytest.mark.parametrize("num_steps", [1, 5, 50])
def test_ode_solver_is_finite_and_reaches_predicted_clean_endpoint(sde_type, num_steps: int) -> None:
    sde = sde_type(device="cpu")
    prediction = torch.tensor([[[[0.5, 0.1]]]], dtype=torch.float64)
    observation = torch.tensor([[[[-0.3, 0.7]]]], dtype=torch.float64)
    solver = sde.get_ode_solver(model_fn=lambda state, time: prediction)

    sample, trajectory, predictions = solver.sampling(observation, num_step=num_steps)

    assert torch.isfinite(sample).all()
    assert torch.isfinite(trajectory).all()
    assert torch.isfinite(predictions).all()
    torch.testing.assert_close(sample, prediction)
