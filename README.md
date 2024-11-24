# Parafoil_landing_with_DRL
Utilizing deep reinforcement learning (DRL) to achieve the precision landing of an autonomous parafoil system.

Specifically, the policy network is trained by the PPO algorithm, and the 6-DOF parafoil dynamics are leveraged.

## Paper
"Precision landing of autonomous parafoil system via deep reinforcement learning", IEEE Aerospace Conference. [Link](https://ieeexplore.ieee.org/document/10521056)

## Result
The DRL agent successfully enables precise terminal landing with a mean touchdown error smaller than 100m.
<div align="center">
  <table style="border: none;">
    <tr>
      <td><img src="https://github.com/Ceaser626/Parafoil_landing_with_DRL/blob/main/figure/Figure_4a.png?raw=true" alt="Figure_4a" width="90%" style="display: inline-block;"/></td>
      <td><img src="https://github.com/Ceaser626/Parafoil_landing_with_DRL/blob/main/figure/Figure_4b.png?raw=true" alt="Figure_4b" width="100%" style="display: inline-block;"/></td>
    </tr>
  </table>
</div>
