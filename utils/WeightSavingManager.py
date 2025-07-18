import os


class WeightSavingManager:

    def __init__(self, game, weight_root_dir="weights"):
        self.weight_root_dir: str = weight_root_dir
        self.model_nro: int = 0
        self.game: str = game

        self.__init_root_dir()

        self.run_nro: int = self.__run_count()
        # new run
        if self.model_nro == 0:
            self.make_new_run_dir()


    def __init_root_dir(self):
        if not self.__root_exists():
            os.mkdir(self.weight_root_dir)


    def __root_exists(self) -> bool:
        return os.path.exists(self.weight_root_dir)
    

    def __run_count(self) -> int:
        return len(os.listdir(self.weight_root_dir))
    

    def get_run_path(self):
        return os.path.join(self.weight_root_dir, f"run{self.run_nro}")
    

    def make_new_run_pathname(self) -> str:
        return os.path.join(
            self.weight_root_dir,
            f"run{self.__run_count()}"
            )
        

    def make_new_run_dir(self) -> str:
        new_run_path = self.make_new_run_pathname()
        
        try:
            os.mkdir(new_run_path)
            return new_run_path
        
        except OSError as e:
            exit(f"Tried to make new directories with path {new_run_path}, \
                 which already exists. Error {e} \nExiting.")
            

    def __make_name(self):
        return f"model_{self.model_nro}_{self.game}.pt"


    def make_save_name(self):
        return os.path.join(self.get_run_path, self.__make_name())