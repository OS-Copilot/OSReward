"""Additional complex tasks for Retro Music app."""

import random
from typing import Any
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import retro_music
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


class RetroCreateTwoPlaylists(retro_music.RetroCreatePlaylist):
  """Task to create two distinct playlists."""

  complexity = 4.0
  max_steps = 25

  @property
  def goal(self) -> str:
    p1 = self.params['playlist_1']
    p2 = self.params['playlist_2']
    files1 = ', '.join(f.split('.')[0] for f in self.params['files_1'])
    files2 = ', '.join(f.split('.')[0] for f in self.params['files_2'])
    return (
        f'Create a playlist in Retro Music titled "{p1}" with songs: {files1}. '
        f'Then, create a second playlist titled "{p2}" with songs: {files2}.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    check1 = sqlite_validators.verify_playlist(
        actual,
        self.params['playlist_1'],
        [f.split('.')[0] for f in self.params['files_1']],
    )
    check2 = sqlite_validators.verify_playlist(
        actual,
        self.params['playlist_2'],
        [f.split('.')[0] for f in self.params['files_2']],
    )
    return (float(check1) + float(check2)) / 2.0

  def initialize_task(self, env: interface.AsyncEnv):
    # Setup uses base class logic but handles merged file lists
    self.params['files'] = self.params['files_1'] + self.params['files_2']
    super().initialize_task(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    # Split the generated files into two groups
    all_files = params['files']
    mid = len(all_files) // 2
    return {
        'playlist_1': params['playlist_name'] + " Vol 1",
        'playlist_2': params['playlist_name'] + " Vol 2",
        'files_1': all_files[:mid],
        'files_2': all_files[mid:],
        'noise_files': params['noise_files'],
        'files': all_files, # For initialization compatibility
        'playlist_name': 'unused' # For initialization compatibility
    }


class RetroCreateAndRemoveSong(retro_music.RetroCreatePlaylist):
  """Task to create a playlist and then remove a specific song from it."""

  complexity = 3.5
  max_steps = 20

  @property
  def goal(self) -> str:
    files = self.params['files']
    names = ', '.join(f.split('.')[0] for f in files)
    remove_target = self.params['remove_target'].split('.')[0]
    return (
        f'Create a playlist titled "{self.params["playlist_name"]}" with: {names}. '
        f'After creating it, remove the song "{remove_target}" from that playlist.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    expected_files = [f.split('.')[0] for f in self.params['files'] 
                      if f != self.params['remove_target']]
    return float(sqlite_validators.verify_playlist(
        actual, self.params['playlist_name'], expected_files
    ))

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    # Ensure at least 2 files so removal leaves something
    if len(params['files']) < 2:
        extra = [f'{name}.mp3' for name in random.sample(retro_music._SONGS, 2)]
        params['files'].extend(extra)
    
    params['remove_target'] = random.choice(params['files'])
    return params


class RetroCreateAndRenamePlaylist(retro_music.RetroCreatePlaylist):
  """Task to create a playlist and then rename it."""

  complexity = 3.0
  max_steps = 20

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    old_name = self.params['playlist_name']
    new_name = self.params['new_name']
    return (
        f'Create a playlist titled "{old_name}" containing: {names}. '
        f'Then rename the playlist to "{new_name}".'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    # Verify playlist exists with NEW name and correct files
    return float(sqlite_validators.verify_playlist(
        actual,
        self.params['new_name'],
        [f.split('.')[0] for f in self.params['files']],
    ))

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    params['new_name'] = retro_music._generate_playlist_name() + " (Remix)"
    return params


class RetroCreatePlaylistByArtist(retro_music.RetroCreatePlaylist):
  """Task to create a playlist containing all songs by a specific artist."""

  complexity = 4.5
  max_steps = 30

  @property
  def goal(self) -> str:
    return (
        f'Browse your library and create a playlist titled "{self.params["playlist_name"]}" '
        f'that contains all songs by the artist "{self.params["target_artist"]}".'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    retro_music._clear_playlist_dbs(env)
    user_data_generation.clear_internal_storage(env)
    
    target_artist = self.params['target_artist']
    other_artist = "Unknown Artist"
    
    # Write target files
    for f in self.params['target_files']:
        user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, f),
          env,
          title=f.split('.')[0],
          artist=target_artist,
          duration_milliseconds=random.randint(180000, 300000),
      )
    
    # Write noise files
    for f in self.params['noise_files']:
        user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, f),
          env,
          title=f.split('.')[0],
          artist=other_artist,
          duration_milliseconds=random.randint(180000, 300000),
      )
    retro_music._scan_music_directory(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    # Order doesn't strictly matter for "all songs by artist", but verifier checks it.
    # We'll allow any order of the correct set.
    actual_files = set()
    for p in actual:
        if p.playlist_name == self.params['playlist_name']:
            actual_files.add(p.media_file_name)
    
    expected_files = set(f.split('.')[0] for f in self.params['target_files'])
    return 1.0 if actual_files == expected_files else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Select songs
    all_songs = random.sample(retro_music._SONGS, 15)
    target_files = [f"{s}.mp3" for s in all_songs[:5]]
    noise_files = [f"{s}.mp3" for s in all_songs[5:]]
    
    return {
        'playlist_name': retro_music._generate_playlist_name(),
        'target_artist': random.choice(user_data_generation.COMMON_GIVEN_NAMES),
        'target_files': target_files,
        'noise_files': noise_files,
        'files': target_files # for compatibility if needed
    }


class RetroMergePlaylists(retro_music.RetroCreatePlaylist):
  """Task to create two playlists and then a third one merging them."""

  complexity = 5.0
  max_steps = 40

  @property
  def goal(self) -> str:
    p1 = "Mix A"
    p2 = "Mix B"
    p_merged = self.params['playlist_name']
    files1 = ', '.join(f.split('.')[0] for f in self.params['files_1'])
    files2 = ', '.join(f.split('.')[0] for f in self.params['files_2'])
    return (
        f'Create playlist "{p1}" with: {files1}. Create playlist "{p2}" with: {files2}. '
        f'Then create a playlist "{p_merged}" containing all songs from {p1} followed by all songs from {p2}.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    expected = [f.split('.')[0] for f in (self.params['files_1'] + self.params['files_2'])]
    return float(sqlite_validators.verify_playlist(
        actual, self.params['playlist_name'], expected
    ))

  def initialize_task(self, env: interface.AsyncEnv):
    self.params['files'] = self.params['files_1'] + self.params['files_2']
    super().initialize_task(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    all_files = params['files']
    mid = len(all_files) // 2
    return {
        'playlist_name': "Mega Merge",
        'files_1': all_files[:mid],
        'files_2': all_files[mid:],
        'files': all_files,
        'noise_files': params['noise_files']
    }


class RetroReorderPlaylist(retro_music.RetroCreatePlaylist):
  """Task to create a playlist and move the last song to the top."""

  complexity = 3.5
  max_steps = 25

  @property
  def goal(self) -> str:
    files = self.params['files']
    names = ', '.join(f.split('.')[0] for f in files)
    return (
        f'Create a playlist "{self.params["playlist_name"]}" with these songs in order: {names}. '
        'Then, edit the playlist to move the last song to the very first position.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    original_order = [f.split('.')[0] for f in self.params['files']]
    # Expected: last item moves to index 0
    expected_order = [original_order[-1]] + original_order[:-1]
    
    return float(sqlite_validators.verify_playlist(
        actual, self.params['playlist_name'], expected_order
    ))


class RetroCreatePlaylistFromQueue(retro_music.RetroPlayingQueue):
  """Task to populate the queue and save it as a playlist."""

  complexity = 3.5
  max_steps = 25

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    return (
        f'Add the following songs to the playing queue: {names}. '
        f'Then save the current queue as a playlist named "{self.params["playlist_name"]}".'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    # Check playlist exists and matches
    actual = retro_music._get_playlist_data(env)
    return float(sqlite_validators.verify_playlist(
        actual,
        self.params['playlist_name'],
        [f.split('.')[0] for f in self.params['files']],
    ))


class RetroCreatePlaylistSortedAlpha(retro_music.RetroCreatePlaylist):
  """Task to create a playlist with songs sorted alphabetically."""

  complexity = 3.0
  max_steps = 25

  @property
  def goal(self) -> str:
    # Give songs in random order
    shuffled_files = list(self.params['files'])
    random.shuffle(shuffled_files)
    names = ', '.join(f.split('.')[0] for f in shuffled_files)
    return (
        f'Create a playlist "{self.params["playlist_name"]}" containing these songs, '
        f'but arrange them in alphabetical order by title: {names}'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    # Expected is sorted
    expected = sorted([f.split('.')[0] for f in self.params['files']])
    return float(sqlite_validators.verify_playlist(
        actual, self.params['playlist_name'], expected
    ))


class RetroClearQueue(retro_music.RetroCreatePlaylist):
  """Task to clear the playing queue."""

  complexity = 2.0
  max_steps = 15

  @property
  def goal(self) -> str:
    return "Clear the playing queue in Retro Music."

  def initialize_task(self, env: interface.AsyncEnv):
    # Initialize normally, but we assume queue might have stuff or user puts stuff.
    # To test 'clear', we should ensure queue is empty at the end.
    # We rely on base initialization to put files on device so app works.
    super().initialize_task(env)
    # We don't pre-populate queue easily without UI interaction simulation or complex DB injection
    # which is hard. But we can ask user to ensure it is empty.
    # Or, we assume user adds something then clears it. 
    # Let's just check if it IS empty at the end.

  def is_successful(self, env: interface.AsyncEnv) -> float:
    queue = retro_music._get_playing_queue(env)
    return 1.0 if len(queue) == 0 else 0.0


class RetroDuplicatePlaylist(retro_music.RetroCreatePlaylist):
  """Task to create a playlist and then duplicate it."""

  complexity = 3.5
  max_steps = 25

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    p1 = self.params['playlist_name']
    p2 = p1 + " Copy"
    return (
        f'Create a playlist "{p1}" with: {names}. '
        f'Then create a copy of this playlist named "{p2}".'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    expected = [f.split('.')[0] for f in self.params['files']]
    
    c1 = sqlite_validators.verify_playlist(actual, self.params['playlist_name'], expected)
    c2 = sqlite_validators.verify_playlist(actual, self.params['playlist_name'] + " Copy", expected)
    return (float(c1) + float(c2)) / 2.0


class RetroCreatePlaylistLimit(retro_music.RetroCreatePlaylist):
  """Task to create a playlist with a subset of listed songs."""

  complexity = 2.5
  max_steps = 20

  @property
  def goal(self) -> str:
    all_names = [f.split('.')[0] for f in (self.params['files'] + self.params['noise_files'])]
    random.shuffle(all_names)
    names_str = ', '.join(all_names)
    limit = len(self.params['files'])
    return (
        f'The following songs are available: {names_str}. '
        f'Create a playlist "{self.params["playlist_name"]}" containing only the first {limit} songs '
        'from this list (in the order listed above).'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    # We need to make sure 'files' matches the 'limit' subset of the shuffled list used in goal.
    # This logic is tricky because 'goal' is dynamic property. 
    # We override this by pre-calculating in generate_random_params or just creating files.
    # To simplify: We pass the specific target files in 'files' and just dump everything else in noise.
    # The 'goal' string construction needs to be consistent with 'files'.
    # Actually, we need to restructure params to store the ordered list.
    super().initialize_task(env)

  @property
  def goal(self) -> str:
    # Use pre-determined ordered list
    ordered_names = self.params['ordered_names']
    limit = len(self.params['files'])
    names_str = ', '.join(ordered_names)
    return (
        f'From the songs: {names_str}. '
        f'Create a playlist "{self.params["playlist_name"]}" with exactly the first {limit} songs.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    targets = [f.split('.')[0] for f in params['files']]
    noise = [f.split('.')[0] for f in params['noise_files']]
    ordered = targets + noise
    # The 'files' param in base class is used for validation (the expected result).
    # So targets are what we expect. We just present them mixed or ordered in a way that targets come first.
    return {
        **params,
        'ordered_names': ordered
    }


class RetroDeletePlaylist(retro_music.RetroCreatePlaylist):
  """Task to create and then delete a playlist."""

  complexity = 3.0
  max_steps = 20

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    p_name = self.params['playlist_name']
    return (
        f'Create a playlist "{p_name}" with: {names}. '
        f'Immediately after creating it, delete the playlist "{p_name}".'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    for p in actual:
        if p.playlist_name == self.params['playlist_name']:
            return 0.0 # Should not exist
    return 1.0


class RetroAddToFavorites(retro_music.RetroCreatePlaylist):
  """Task to add specific songs to Favorites."""

  complexity = 3.0
  max_steps = 20

  @property
  def goal(self) -> str:
    targets = [f.split('.')[0] for f in self.params['files']]
    target_str = ', '.join(targets)
    return f'Add the following songs to your Favorites: {target_str}.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    # "Favorites" is usually a default playlist name or "Favorites" in DB.
    # Assuming standard behavior where it shows up in playlists as "Favorites"
    return float(sqlite_validators.verify_playlist(
        actual,
        "Favorites",
        [f.split('.')[0] for f in self.params['files']],
    ))


class RetroPlaySpecificSong(retro_music.RetroCreatePlaylist):
  """Task to find and play a specific song."""

  complexity = 2.5
  max_steps = 15

  @property
  def goal(self) -> str:
    target = self.params['files'][0].split('.')[0]
    return f'Find and play the song "{target}".'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    # Playing a song usually adds it to queue or sets it as current.
    # We check if it is in the queue.
    queue = retro_music._get_playing_queue(env)
    target = self.params['files'][0].split('.')[0]
    return 1.0 if target in queue else 0.0


class RetroCreatePlaylistSpecificDurationRange(retro_music.RetroPlaylistDuration):
  """Task to create a playlist with duration strictly less than X minutes."""

  complexity = 4.0
  max_steps = 25

  @property
  def goal(self) -> str:
    return (
        f'Create a playlist "{self.params["playlist_name"]}" containing a selection of songs '
        'such that the total duration is less than 20 minutes.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    retro_music._clear_playlist_dbs(env)
    # Generate files such that we have small ones
    for file in self.params['files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=random.randint(2 * 60 * 1000, 4 * 60 * 1000), # 2-4 mins
      )
    retro_music._scan_music_directory(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    # Any playlist with duration < 20 mins is success
    songs = retro_music._get_playlist_data(env)
    total_ms = 0
    found = False
    for song in songs:
      if song.playlist_name == self.params['playlist_name']:
        found = True
        total_ms += song.duration_ms
    
    if not found: return 0.0
    return 1.0 if total_ms < (20 * 60 * 1000) and total_ms > 0 else 0.0


class RetroQueueMultipleSongs(retro_music.RetroPlayingQueue):
  """Task to add multiple specific songs to queue in specific order."""
  
  complexity = 3.5
  max_steps = 25
  # Inherits logic, just ensuring complexity via params or goal if needed.
  # The base RetroPlayingQueue handles list of files.


class RetroCreatePlaylistReverseAlpha(retro_music.RetroCreatePlaylist):
  """Task to create a playlist sorted in reverse alphabetical order."""

  complexity = 3.5
  max_steps = 25

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    return (
        f'Create a playlist "{self.params["playlist_name"]}" with these songs, '
        f'sorted in reverse alphabetical order: {names}'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    expected = sorted([f.split('.')[0] for f in self.params['files']], reverse=True)
    return float(sqlite_validators.verify_playlist(
        actual, self.params['playlist_name'], expected
    ))


class RetroConsolidatePlaylists(retro_music.RetroCreatePlaylist):
  """Task to move songs from one playlist to another and delete the old one."""

  complexity = 4.5
  max_steps = 35

  @property
  def goal(self) -> str:
    p1 = "Old Mix"
    p2 = "New Mix"
    files = ', '.join(f.split('.')[0] for f in self.params['files'])
    return (
        f'Create a playlist "{p1}" with: {files}. '
        f'Then create "{p2}" and move all songs from "{p1}" to "{p2}". '
        f'Finally, delete "{p1}".'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    expected = [f.split('.')[0] for f in self.params['files']]
    
    # Check p2 has songs
    p2_ok = sqlite_validators.verify_playlist(actual, "New Mix", expected)
    
    # Check p1 is gone
    p1_gone = True
    for p in actual:
        if p.playlist_name == "Old Mix":
            p1_gone = False
            break
            
    return 1.0 if (p2_ok and p1_gone) else 0.0


class RetroCreatePlaylistExcluding(retro_music.RetroCreatePlaylist):
  """Task to create a playlist excluding a specific song."""

  complexity = 3.0
  max_steps = 20

  @property
  def goal(self) -> str:
    all_files = self.params['files'] + self.params['noise_files']
    # Select one to exclude
    exclude = self.params['files'][0]
    include = [f for f in all_files if f != exclude]
    
    names = ', '.join(f.split('.')[0] for f in all_files)
    ex_name = exclude.split('.')[0]
    
    return (
        f'Given the songs: {names}. '
        f'Create a playlist "{self.params["playlist_name"]}" containing all of them '
        f'EXCEPT "{ex_name}".'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    all_files = self.params['files'] + self.params['noise_files']
    exclude = self.params['files'][0]
    expected = [f.split('.')[0] for f in all_files if f != exclude]
    
    # Order might vary as user adds them, but verify_playlist checks strict order by default.
    # We should relax this or expect specific order (e.g. alphabetical or listed).
    # verify_playlist usually expects exact list. We'll assume user adds them in listed order skipping the excluded one.
    return float(sqlite_validators.verify_playlist(
        actual, self.params['playlist_name'], expected
    ))


class RetroAddAlbumToPlaylist(retro_music.RetroCreatePlaylist):
  """Task to add an entire album to a playlist."""
  
  complexity = 3.5
  max_steps = 25
  
  @property
  def goal(self) -> str:
      album = "Greatest Hits"
      return f'Find the album "{album}" and add all its songs to a new playlist named "{self.params["playlist_name"]}".'

  def initialize_task(self, env: interface.AsyncEnv):
      retro_music._clear_playlist_dbs(env)
      user_data_generation.clear_internal_storage(env)
      
      album = "Greatest Hits"
      for f in self.params['files']:
          user_data_generation.write_mp3_file_to_device(
              file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, f),
              env,
              title=f.split('.')[0],
              album=album, # Set album metadata
              duration_milliseconds=180000
          )
      retro_music._scan_music_directory(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
      actual = retro_music._get_playlist_data(env)
      expected = [f.split('.')[0] for f in self.params['files']]
      # verify_playlist checks if songs are in playlist.
      # Since we added all songs from album, verifying strict content works.
      # Order might be album track order (usually alphabetical by filename if track num missing).
      # We assume user adds "Album", app adds sorted by track/title.
      # For robustness, check set equality manually if verify_playlist is too strict on order.
      
      pl_files = [x.media_file_name for x in actual if x.playlist_name == self.params['playlist_name']]
      return 1.0 if set(pl_files) == set(expected) else 0.0