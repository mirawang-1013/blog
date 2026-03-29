import { getAllPosts } from "@/lib/posts";

export default function Home() {
  const posts = getAllPosts();

  return (
    <div>
      <section className="mb-16">
        <h1 className="text-4xl font-bold mb-4">Hi, I'm Hipvan</h1>
        <p className="text-[var(--color-muted)] text-lg leading-relaxed">
          NUS CS grad exploring AI, building things, and writing about what I
          learn. This blog is powered by an Obsidian Agent I built myself.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-8">Posts</h2>
        {posts.length === 0 ? (
          <p className="text-[var(--color-muted)]">Coming soon...</p>
        ) : (
          <div className="space-y-8">
            {posts.map((post) => (
              <article key={post.slug} className="group">
                <a href={`/blog/${post.slug}`} className="block">
                  <time className="text-sm text-[var(--color-muted)]">
                    {post.date}
                  </time>
                  <h3 className="text-xl font-semibold mt-1 group-hover:text-[var(--color-accent)] transition-colors">
                    {post.title}
                  </h3>
                  {post.summary && (
                    <p className="text-[var(--color-muted)] mt-2">
                      {post.summary}
                    </p>
                  )}
                  {post.tags.length > 0 && (
                    <div className="flex gap-2 mt-3">
                      {post.tags.map((tag) => (
                        <span
                          key={tag}
                          className="text-xs px-2 py-1 rounded-full bg-[var(--color-card)] text-[var(--color-muted)]"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </a>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
