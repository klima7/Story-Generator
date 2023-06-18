from lightning.pytorch.callbacks import Callback


class GenerateCallback(Callback):
    
    def __init__(self, prompt, temperatures=(0.1,), length=100, interval=1_000):
        super().__init__()
        
        self.prompt = prompt
        self.temperatures = temperatures
        self.length = length
        self.interval = interval
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step % self.interval == 0:
            self.generate(pl_module)
            
    def generate(self, model):
        tensorboard = model.logger.experiment
        for temperature in self.temperatures:
            text = model.generate(self.prompt, length=self.length, temperature=temperature)
            tensorboard.add_text(f'generate_{temperature}', text, global_step=model.global_step)
