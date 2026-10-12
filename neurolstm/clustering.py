"""EV-k-Means (2.25)-(2.30): точно эквивалентное взвешенное евклидово пространство."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class EVKMeans:
    def __init__(self, clusters: int=3, components: int=8,
                 weights: tuple[float,float,float] | list[float]=(0.5,0,0.5),
                 epsilon: float=1e-8, seed: int=42, restarts: int=10):
        self.clusters,self.components,self.epsilon,self.seed,self.restarts=clusters,components,epsilon,seed,restarts
        w=np.asarray(weights,float)
        if w.shape!=(3,) or (w<0).any() or not np.isclose(w.sum(),1):
            raise ValueError("Нужны три неотрицательных веса с суммой 1")
        self.weights=w

    def _blocks(self, x: np.ndarray, modal: np.ndarray | None, physical: np.ndarray, fit: bool) -> list[np.ndarray]:
        if fit:
            self.pca=PCA(n_components=min(self.components,x.shape[1],len(x)-1),svd_solver="full")
            z=self.pca.fit_transform(x)
        else:
            z=self.pca.transform(x)
        z=z/np.sqrt((self.pca.explained_variance_+self.epsilon)*z.shape[1])
        if modal is None:
            if self.weights[1]>0:
                raise ValueError("Нет модальных дескрипторов, но omega_2>0")
            modal=np.zeros((len(x),1))
        blocks=[z,modal/np.sqrt(modal.shape[1]),physical/np.sqrt(physical.shape[1])]
        if fit:
            self.scales=np.array([max(float(np.mean(np.sum((b-b.mean(0))**2,axis=1))),self.epsilon) for b in blocks])
        return blocks

    def fit_predict(self, numeric: np.ndarray, physical: np.ndarray, modal: np.ndarray | None=None) -> np.ndarray:
        if self.clusters<1 or self.clusters>len(numeric):
            raise ValueError("Число кластеров несовместимо с размером train")
        blocks=self._blocks(numeric,modal,physical,True)
        transformed=np.concatenate([b*np.sqrt(w/s) for b,w,s in zip(blocks,self.weights,self.scales)],axis=1)
        self.model=KMeans(n_clusters=self.clusters,n_init=self.restarts,random_state=self.seed)
        labels=self.model.fit_predict(transformed)
        self.training_cvr=1/(1+self.model.inertia_/len(numeric))
        return labels

    def predict(self, numeric: np.ndarray, physical: np.ndarray, modal: np.ndarray | None=None) -> np.ndarray:
        blocks=self._blocks(numeric,modal,physical,False)
        x=np.concatenate([b*np.sqrt(w/s) for b,w,s in zip(blocks,self.weights,self.scales)],axis=1)
        return self.model.predict(x)


def clustered_order(labels: np.ndarray, seed: int) -> list[int]:
    """Все окна ровно один раз, однородные по кластеру серии; без oversampling."""
    rng=np.random.default_rng(seed)
    groups=[rng.permutation(np.flatnonzero(labels==label)) for label in np.unique(labels)]
    rng.shuffle(groups)
    return np.concatenate(groups).tolist()
