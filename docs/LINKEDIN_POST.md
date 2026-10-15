# LinkedIn draft

Notes for the author (not part of the post):

- Upload `media/demo.mp4` natively to LinkedIn (not as a link); it is 57 s long.
- Replace `<REPO_URL>` with the repository link.
- Sources of every number: "10 de 10 / 10 of 10" → `results/benchmark.md` (adversarial blocked, also
  `pytest tests/test_safety.py`); "0 contatos em 60 episódios / 0 contacts in 60 episodes" →
  `results/benchmark.md` (20 seeds × 3 detectors, `cobotsort benchmark --seeds 20 --detector oracle hsv yolo
  --provider mock`); "0,991 vs 0,942 / 0.991 vs 0.942" → `results/perception.md` (`cobotsort eval-vision`,
  mAP@50 on the hard-lighting set).

---

## Português

Construí uma célula robótica que separa peças por cor e bloqueia 10 de 10 comandos inseguros.

O CobotSort AI junta um cobot de 6 eixos com ventosa (um UR5e no MuJoCo), um detector YOLO11n treinado só com imagens sintéticas e um agente que recebe ordens em português ou inglês.

Três números, todos gerados por scripts do repositório:
- 0 contatos não planejados em 60 episódios de separação automática
- mAP@50 de 0,991 sob uma iluminação que o modelo nunca viu no treino (o baseline clássico em HSV fica em 0,942)
- 10 de 10 pedidos inseguros bloqueados antes de qualquer movimento

O BRICS Hi-Tech Skills Camp me ensinou a ordem certa: segurança, depois automação, precisão e, por último, velocidade. O benchmark do projeto segue exatamente essa ordem.

O agente usa o mesmo padrão do EduBot: o modelo decide o quê; código determinístico decide como e se é seguro. Ele não muda velocidades nem limites e, se a API falhar, um planejador por regras assume.

É uma simulação: nenhum resultado vem de hardware real.

Código e vídeo: <REPO_URL>

#Robótica #VisãoComputacional #IAIndustrial #Cobots #MuJoCo

---

## English

I built a robotic cell that sorts parts by color and blocks 10 out of 10 unsafe commands.

CobotSort AI combines a 6-axis cobot with a vacuum gripper (a UR5e in MuJoCo), a YOLO11n detector trained only on synthetic images, and an agent that takes work orders in Portuguese or English.

Three numbers, all produced by scripts in the repo:
- 0 unplanned contacts in 60 automatic sorting episodes
- mAP@50 of 0.991 under lighting the model never saw in training (the classical HSV baseline gets 0.942)
- 10 out of 10 unsafe requests blocked before any motion

The BRICS Hi-Tech Skills Camp taught me the right order: safety first, then automation, precision and, last, speed. The project's benchmark follows exactly that order.

The agent reuses the EduBot pattern: the model decides what; deterministic code decides how and whether it is safe. It cannot change speeds or limits, and if the API fails a rule-based planner takes over.

This is a simulation: no result comes from real hardware.

Code and video: <REPO_URL>

#Robotics #ComputerVision #IndustrialAI #Cobots #MuJoCo
