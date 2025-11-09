-- Create storage buckets for OpenManus-RL

-- Create bucket for datasets
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'datasets',
  'datasets',
  false,
  52428800, -- 50MB
  array['application/json', 'text/plain', 'text/csv', 'application/zip']
);

-- Create bucket for model checkpoints
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'model-checkpoints',
  'model-checkpoints',
  false,
  1073741824, -- 1GB
  array['application/octet-stream', 'application/zip', 'application/x-tar']
);

-- Create bucket for logs
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'logs',
  'logs',
  false,
  10485760, -- 10MB
  array['text/plain', 'application/json', 'text/csv']
);

-- Create bucket for evaluation results
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'evaluation-results',
  'evaluation-results',
  false,
  52428800, -- 50MB
  array['application/json', 'text/plain', 'text/csv', 'text/html']
);

-- Storage policies for datasets bucket
create policy "Allow service role to manage datasets"
on storage.objects for all
to service_role
using (bucket_id = 'datasets')
with check (bucket_id = 'datasets');

create policy "Allow authenticated users to read datasets"
on storage.objects for select
to authenticated
using (bucket_id = 'datasets');

-- Storage policies for model-checkpoints bucket
create policy "Allow service role to manage model checkpoints"
on storage.objects for all
to service_role
using (bucket_id = 'model-checkpoints')
with check (bucket_id = 'model-checkpoints');

create policy "Allow authenticated users to read model checkpoints"
on storage.objects for select
to authenticated
using (bucket_id = 'model-checkpoints');

-- Storage policies for logs bucket
create policy "Allow service role to manage logs"
on storage.objects for all
to service_role
using (bucket_id = 'logs')
with check (bucket_id = 'logs');

create policy "Allow authenticated users to read logs"
on storage.objects for select
to authenticated
using (bucket_id = 'logs');

-- Storage policies for evaluation-results bucket
create policy "Allow service role to manage evaluation results"
on storage.objects for all
to service_role
using (bucket_id = 'evaluation-results')
with check (bucket_id = 'evaluation-results');

create policy "Allow authenticated users to read evaluation results"
on storage.objects for select
to authenticated
using (bucket_id = 'evaluation-results');
