-- Supabase schema for Vision-Based PPO Sky-Road Arena.
--
-- Run this in Supabase SQL Editor. Then create a public Storage bucket named
-- "videos" from the Storage UI, or keep the storage policies below if the
-- bucket already exists.

create table if not exists submissions (
  id uuid default gen_random_uuid() primary key,
  creator_name text not null,
  creator_uid text not null,
  tag text not null,
  mean_return double precision,
  mean_episode_length double precision,
  route_completion double precision,
  best_return double precision,
  map_scores jsonb default '{}'::jsonb,
  episode_results jsonb default '[]'::jsonb,
  video_url text,
  created_at timestamptz default now()
);

alter table submissions enable row level security;

drop policy if exists "Anyone can read submissions" on submissions;
drop policy if exists "Anyone can insert submissions" on submissions;

create policy "Anyone can read submissions"
on submissions for select
using (true);

create policy "Anyone can insert submissions"
on submissions for insert
with check (true);

-- Optional: create the public videos bucket through SQL.
-- If it already exists, this does nothing.
insert into storage.buckets (id, name, public)
values ('videos', 'videos', true)
on conflict (id) do nothing;

drop policy if exists "Anyone can read videos" on storage.objects;
drop policy if exists "Anyone can upload videos" on storage.objects;

create policy "Anyone can read videos"
on storage.objects for select
using (bucket_id = 'videos');

create policy "Anyone can upload videos"
on storage.objects for insert
with check (bucket_id = 'videos');
