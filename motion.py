from dataclasses import dataclass
import math, time

PRESETS={
 "idle":{"emotion":"warm","smile":.12,"head_motion":.08,"blink":True},
 "speaking":{"emotion":"warm","smile":.22,"head_motion":.18,"blink":True},
 "asking_question":{"emotion":"curious","smile":.28,"head_motion":.22,"blink":True,"question_expression":True},
 "explaining":{"emotion":"focused","smile":.12,"head_motion":.2,"blink":True},
 "praise":{"emotion":"happy","smile":.55,"head_motion":.3,"blink":True,"nod":True},
 "encouraging":{"emotion":"warm","smile":.42,"head_motion":.2,"blink":True},
 "thinking":{"emotion":"thoughtful","smile":.05,"head_motion":.1,"blink":True},
}

@dataclass
class MotionEngine:
    state: str="idle"; started: float=time.monotonic()
    def set(self,state: str): self.state=state if state in PRESETS else "speaking"; self.started=time.monotonic()
    def parameters(self):
        p=dict(PRESETS[self.state]); t=time.monotonic()-self.started
        p["breathing"]=.5+.5*math.sin(t*1.4); p["blink_phase"]=(t%4.2)<.12
        p["nod_phase"]=math.sin(t*5)*.5+.5 if p.get("nod") else 0
        return p
