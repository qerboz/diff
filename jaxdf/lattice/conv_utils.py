import numpy as np
import jax.numpy as jnp
from typing import Callable, Union
from functools import partial


def conv_indices(shape: tuple[int, ...], return_flat=True, center=True):
    """Generate index matrix for translation invariant layer (conv).

    This can be understood as giving the 'orbit index' for each pair of
    points x, y.

    Returns:
        An array of shape ``[len(shape), *shape, *shape]`` potentially
        flattened to ``[len(shape), prod(shape), prod(shape)]`` where
       ``out[i, x, y] = (y-x)[i]``.
    """
    xs_grid = np.indices(shape)
    flat = np.reshape(xs_grid, (len(shape), np.prod(shape)))
    mods = np.array(shape).reshape((-1, 1, 1))
    if center:
        shift = np.array(shape).reshape((-1, 1, 1)) // 2
    else:
        shift = 0
    added = np.mod(flat[:, None, :] - flat[:, :, None] + shift, mods)
    full_shape = (len(shape), *shape, *shape)
    return added if return_flat else added.reshape(full_shape)


def _lattice_distances(shape: tuple[int, ...]):
    """Distance squared to origin of lattice sites.

    Args:
        shape: Shape of lattice.
    Returns:
        An integer array where each entry is the site's
        distance to the origin squared. The origin is
        at L//2 in each dimension of length L.
    """
    coords = [np.arange(-(s - 1) // 2, (s + 1) // 2) for s in shape]
    coords = np.meshgrid(*coords, indexing='ij')
    dist = sum(c ** 2 for c in coords)
    return dist


def _gather_orbit_indices(orbits: jnp.ndarray):
    """Translate non-contiguous orbit indices to sequential ones."""
    unique = np.unique(orbits)
    index_map = np.empty(unique[-1] + 1, dtype=int)
    index_map[unique] = np.arange(len(unique))
    orbits = index_map[orbits]
    return len(unique), orbits


def unique_index_kernel(shape: tuple[int, ...]):
    """Unique lattice indices ordered by distance to origin.

    Args:
        shape: Shape of lattice.
    Returns:
        An integer array of given `shape` with unique entries ordered
        by the site's distance to the origin at L//2 (in each dimension).
    """
    dist = _lattice_distances(shape)
    index = np.empty(np.prod(shape), dtype=int)
    index[np.argsort(dist.flatten())] = np.arange(np.prod(shape))
    index = index.reshape(shape)
    return index


def flip_lattice(lattice: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Flip array along one axis."""
    return np.roll(np.flip(lattice, axis), 1 - lattice.shape[axis] % 2, axis)


def rot_lattice_90(lattice: jnp.ndarray, ax1: int, ax2: int) -> jnp.ndarray:
    """Rotate two axes into each other by 90 degrees."""
    lattice = np.swapaxes(lattice, ax1, ax2)
    lattice = flip_lattice(lattice, ax1)
    return lattice


def gather_orbits(
        shape: tuple[int, ...],
        transformations: list[Callable[[jnp.ndarray], jnp.ndarray]]) \
        -> tuple[int, jnp.ndarray]:
    """Compute orbit indices for lattice given shape and transformations.

    One orbit is defined by a set of indices which can be transformed
    into each other by any composition of operations given by the
    list of transformations.

    Args:
        shape: Lattice shape.
        transformations: List of transformations.
            Each transformation takes an array as input and returns
            a correspondingly transformed array.
    Returns:
        Tuple (number of orbits, integer array of orbit for each site).
    """
    # start by assigning a unique 'orbit' to each index of the lattice
    lattice = unique_index_kernel(shape)
    unique_indices = lattice.flatten()

    # generate the stack of transformed lattices
    partial_orbits = np.empty(
        (len(unique_indices), len(transformations) + 1), dtype=int)
    partial_orbits[:, 0] = unique_indices
    for i, op in enumerate(transformations):
        partial_orbits[:, i + 1] = op(lattice).flatten()

    # maps orbit -> set(indices)
    orbit_members = dict()
    # maps index -> orbit
    orbit_index = np.full_like(unique_indices, -1)

    # consider all indices that appear at a given site given the
    # 'generator' transformations applied above
    for parts in partial_orbits:
        orbits = set()  # track what orbits need to be merged
        update_indices = set()  # track for which indices orbit needs updating
        for i in parts:
            if orbit_index[i] != -1:  # index was already assigned to an orbit
                orbits.add(orbit_index[i])
            else:
                update_indices.add(i)

        # let the orbit index be the minimum of the site indices it contains
        new_orbit = np.min(parts)
        if len(orbits) != 0:
            new_orbit = min(new_orbit, min(orbits))

        if new_orbit not in orbits:
            members = update_indices.copy()
            orbit_members[new_orbit] = members
        else:
            members = orbit_members[new_orbit]
            members.update(update_indices)

        # merge orbits
        for orbit in orbits:
            if orbit == new_orbit:
                continue
            new_members = orbit_members[orbit]
            members.update(new_members)
            update_indices.update(new_members)
            del orbit_members[orbit]

        for i in update_indices:
            orbit_index[i] = new_orbit

    new_orbits = orbit_index[unique_indices].reshape(lattice.shape)
    return _gather_orbit_indices(new_orbits)


def kernel_d4(shape: tuple[int, ...]) -> tuple[int, jnp.ndarray]:
    """Orbit number and indices per site for D4 symmetry.

    Args:
        shape: Lattice shape (tuple).
    Returns:
        Tuple (number of orbits, integer array of orbit for each site).
    """
    assert all(shape[0] == li for li in shape[1:]), \
        'Rotation requires all side lengths to be equal.'
    transformations = [partial(flip_lattice, axis=i)
                       for i in range(len(shape))]
    for i in range(len(shape)):
        # probably redundantly many
        for j in range(i, len(shape)):
            transformations.append(partial(rot_lattice_90, ax1=i, ax2=j))

    return gather_orbits(shape, transformations)


def kernel_equidist(shape: tuple[int, ...]) -> tuple[int, jnp.ndarray]:
    """Orbit number and indices per site identifying equidistant sites.

    Args:
        shape: Lattice shape (tuple).
    Returns:
        Tuple (number of orbits, integer array of orbit for each site).
    """
    dist = _lattice_distances(shape)
    return _gather_orbit_indices(dist)


def unfold_kernel(
        kernel_params: jnp.ndarray, orbits: jnp.ndarray) -> jnp.ndarray:
    """Expand the parameters of a symmetric kernel into full conv kernel.

    This function is the inverse of ``fold_kernel``.

    Args:
        kernel_params: Parameters of kernel with shape
            (orbit_count, in_channels, out_channels).
        orbits: Integer array giving the orbit index for each lattice site.

    Returns:
        The full convolutional kernel.
    """
    return kernel_params[orbits]


def pad_kernel_weights(
        kernel: jnp.ndarray,
        new_shape: Union[int, tuple[int, ...]]) -> jnp.ndarray:
    """Increase the size of conv kernel by padding with zeros.

    The non-trivial part comes from dimensions with even length.
    Using circular padding, the edge value is copied when going to a
    larger length and divided by the number of copies.

    Args:
        kernel: Kernel weight array (already expanded, not the
            reduced parameters). Shape of array is
            ``(L_1, ..., L_dim, in_channels, out_channels)``.
        new_shape: Either an integer or a tuple.
            New side length of lattice. Must be larger than the
            side length L of the original kernel.

    Returns:
        Kernel with new shape.
    """
    shape = kernel.shape[:-2]
    if isinstance(new_shape, tuple):
        assert len(new_shape) == len(shape), \
            'The dimension of the new shape does not match existing kernel.'
    else:
        new_shape = (new_shape,) * len(shape)

    # in even dimensions, copy the 'wrap-around' indices (at the edge)
    wraps = [(0, 1) if length % 2 == 0 else (0, 0) for length in shape]
    kernel = np.pad(kernel, [*wraps, (0, 0), (0, 0)], 'wrap')
    _slice = np.index_exp[:]
    for dim, length in enumerate(shape):
        if length % 2 == 0:
            kernel[_slice * dim + ([0, -1],)] /= 2

    # add zeros to reach desired shape
    padding = [((new - old + 1) // 2, (new - old) // 2)
               for new, old in zip(new_shape, kernel.shape[:-2])]
    w = np.pad(kernel, padding + [(0, 0)] * 2, 'constant', constant_values=0.)
    return w


def fold_kernel(
        kernel_weights: jnp.ndarray,
        orbits: jnp.ndarray,
        orbit_count: int) -> jnp.ndarray:
    """Convert symmetric conv kernel into the unique independent parameters.

    This function is the inverse of ``unfold_kernel``.

    Args:
        kernel_weights: Kernel weight array of a convolution.
        orbits: Integer array giving the orbit index for each lattice site.
        orbit_count: Total number of orbits.

    Returns:
        Reduced array giving the independent parameters of the conv kernel.
    """
    in_channels, out_channels = kernel_weights.shape[-2:]
    assert orbits.shape == kernel_weights.shape[:-2]

    w_raw = np.zeros((orbit_count, in_channels, out_channels))
    count = np.zeros((orbit_count, 1, 1))

    flat_kernel = kernel_weights.reshape(-1, in_channels, out_channels)
    for index, ws in zip(orbits.flatten(), flat_kernel):
        w_raw[index] += ws
        count[index] += 1

    return w_raw / count
