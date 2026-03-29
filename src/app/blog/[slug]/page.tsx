import { getPostBySlug, getAllSlugs } from "@/lib/posts";
import { notFound } from "next/navigation";

export async function generateStaticParams() {
  return getAllSlugs().map((slug) => ({ slug }));
}

export default async function BlogPost({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  try {
    const post = await getPostBySlug(slug);
    return (
      <article>
        <header className="mb-12">
          <time className="text-sm text-[var(--color-muted)]">{post.date}</time>
          <h1 className="text-4xl font-bold mt-2">{post.title}</h1>
          {post.tags.length > 0 && (
            <div className="flex gap-2 mt-4">
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
        </header>
        <div
          className="prose"
          dangerouslySetInnerHTML={{ __html: post.contentHtml }}
        />
        <footer className="mt-16 pt-8 border-t border-[var(--color-border)]">
          <a
            href="/"
            className="text-[var(--color-accent)] hover:underline"
          >
            &larr; Back to all posts
          </a>
        </footer>
      </article>
    );
  } catch {
    notFound();
  }
}
